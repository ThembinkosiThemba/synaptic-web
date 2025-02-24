use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    f32,
    time::{Instant, UNIX_EPOCH},
};

use egui::{vec2, Color32, Key, Painter, PointerButton, Pos2, Rect, Stroke, Ui, ViewportBuilder};
use nalgebra as na;
use noise::{NoiseFn, OpenSimplex};
use rand::{random, rng, Rng};

const REPRODUCING_INTERVAL: f32 = 10.0; // Every 2 seconds
const HEALTH_DECAY_RATE: f32 = 0.0001; // decay rate at (0.01% per second)
const SPHERE_SEGMENTS: usize = 16; // 3D sphere rendering
const NODE_SEPARATION_FORCE: f32 = 1.0; // force to keep nodes apart
const MAX_REPRODUCING_ATTEMPTS: u32 = 3000; // Maximum number of times a node can reproduce
                                            // const HEALTH_DECAY_RATE: f32 = 0.1; // Health reduction per second
const HALF_LIFE: f32 = 1_000_000_.0; // Seconds until health is halved

pub const HASHMAP_DIRECT_LOOKUP: &str = "HashMap Direct Lookup";
pub const LINEAR_SEARCH: &str = "Linear Search";
pub const BFS: &str = "Breath-First Search";
pub const DSP: &str = " Dijkstra's Shortest Path";

#[derive(Clone)]
struct SearchMetrics {
    algorithm: String, // algorithm represents the algorithm that's going to be used for searching a particular node in the system
    comparisons: u64, // The "comparisons" field in SearchMetrics counts how many times the algorithm
    // needed to compare values to find the target. It's a measure of algorithmic
    //efficiency independent of time (which can vary based on hardware).
    duration_ns: u128, // time taken to execute search
    big_o: String,     // time complexity for the algorithm
    space_complexity: String,
}

// SearchAlgorithms contains the different algorithms for finding a particular node on the ecosystem
#[derive(PartialEq)]
enum SearchAlgorithm {
    //    HashMap
    //    - Best for: Quick direct access when ID is known
    //    - How it works: Uses HashMap's internal hash function to directly access the node
    //    - Time Complexity: O(1) & Space Complexity: O(1)
    //    - Comparisons: Only 1 comparison needed
    //    - Pros: Fastest for direct lookups into the data structure
    //    - Cons: Doesn't utilize network structure
    HashMap,

    //    Linear
    //    - Best for: Small datasets or when HashMap isn't available
    //    - How it works: Iterates through all nodes sequentially
    //    - Time Complexity: O(n) & Space Complexity: O(1)
    //    - Comparisons: Up to n comparisons (n = number of nodes)
    //    - Pros: Simple implementation, no extra space needed
    //    - Cons: Slow for large datasets
    Linear,

    //    Breadth-First Search
    //    - Best for: Finding nodes by traversing relationships
    //    - How it works: Explores network level by level starting from root
    //    - Time Complexity: O(V + E) where V = vertices, E = edges & Space Complexity: O(V)
    //    - Comparisons: Depends on node position in network
    //    - Pros: Can find related nodes efficiently
    //    - Cons: More complex, uses more memory
    BFS,

    DSP, // TODO: documentation

    // Comparison for all algos
    CompareAll,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct State {
    cost: i32,
    node: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// NodeProperties holds the properties for each node
// it has the following properties
#[derive(Clone)]
#[allow(dead_code)]
struct NodeProperties {
    color: [f32; 3],      // color of the node
    size: f32,            // size of the node
    strength: f32,        // strength the node possesses
    health: f32,          // Current health of the node
    half_life: f32,       // Time until health is halved
    reproduction: u32,    // Number of times this node has reproduced
    generation: u32,      // Which generation this node belongs to
    learning_factor: f32, // preference for pairing with high-quality nodes (0.0 to 1.0)
}

// Node defines the structure of a single node in the system
#[derive(Clone)]
struct Node {
    id: usize,                       // id of the node, which will be used for identification
    position: na::Point3<f32>, // position of the node in the system. It uses Point3 which is a statically sized 3-dimensional column point.
    velocity: na::Vector3<f32>, // velocity of the node, since by default, the nodes are not fixed on one point. It uses A stack-allocated, 3-dimensional column vector.
    properties: NodeProperties, // ref to NodeProperties
    parents: Option<(usize, usize)>, // Each node, more expectially child nodes should have parent nodes
}

// `SynapticWeb` represents a complex network/graph where:
// - Nodes are stored in a `HashMap<usize, Node>`
// - Each node has a unique ID (key in HashMap)
// - Nodes have parent relationships (creating a directed graph)
// - The structure combines properties of both:
//   1. A Graph (nodes can have multiple connections)
//   2. A Tree (nodes have generational hierarchy)
// This hybrid structure makes it interesting for different search algorithms, each with their own trade-offs.
struct SynapticWeb {
    nodes: HashMap<usize, Node>,
    next_id: usize,
    view: View,
    selected_node: Option<usize>,
    last_reproduce_time: Instant,
    highlighted_path: Option<(Vec<usize>, Color32)>,
    search_id: String,
    search_results: Vec<SearchMetrics>,
    selected_algo: SearchAlgorithm,
    avg_health_history: Vec<(f32, f32)>, //(time, avg_health) for evolution training
}

// View structure for 3D navigation
// It controls the view of the entire eco-system
struct View {
    position: na::Point3<f32>,
    rotation: na::UnitQuaternion<f32>,
    zoom: f32,
}

impl SynapticWeb {
    fn new() -> Self {
        let mut registry = Self {
            nodes: HashMap::new(),
            next_id: 0, // next_id 0 will represent the inception node
            view: View::default(),
            selected_node: None,
            last_reproduce_time: Instant::now(),
            highlighted_path: None,
            search_id: String::new(),
            search_results: Vec::new(),
            selected_algo: SearchAlgorithm::HashMap,
            avg_health_history: Vec::new(),
        };

        // creation of the initial parent node (world inception) and addition of the initial children from the parent
        // node
        registry.add_inception_node();
        registry.add_initial_children();

        registry
    }

    // draw_3d_node is used to draw the shapes of the nodes on the ecosystem
    fn draw_3d_node(&self, painter: &Painter, center: Pos2, radius: f32, color: Color32) {
        let node = self.nodes.values().next().unwrap();

        let base_color = Color32::from_rgb(
            (node.properties.color[0] * 255.0) as u8,
            (node.properties.color[1] * 255.0) as u8,
            (node.properties.color[2] * 255.0) as u8,
        );

        let dead_color = Color32::GRAY;
        let health = node.properties.health;
        let effective_color = Color32::from_rgb(
            (base_color.r() as f32 * health + dead_color.r() as f32 * (1.0 - health)) as u8,
            (base_color.g() as f32 * health + dead_color.g() as f32 * (1.0 - health)) as u8,
            (base_color.b() as f32 * health + dead_color.b() as f32 * (1.0 - health)) as u8,
        );

        let mut points = Vec::new();

        // Generating sphere points
        for i in 0..SPHERE_SEGMENTS {
            let lat = std::f32::consts::PI * (i as f32) / (SPHERE_SEGMENTS as f32);
            for j in 0..SPHERE_SEGMENTS {
                let lon = 2.0 * std::f32::consts::PI * (j as f32) / (SPHERE_SEGMENTS as f32);

                let x = radius * lat.sin() * lon.cos();
                let y = radius * lat.sin() * lon.sin();
                let z = radius * lat.cos();

                points.push(egui::Pos2::new(
                    center.x + x,
                    center.y + y * (0.5 + 0.5 * z / radius), // Perspective scaling
                ));
            }
        }

        // Draw sphere segments
        for i in 0..points.len() {
            let next_i = (i + 1) % points.len();
            painter.line_segment(
                [points[i], points[next_i]],
                Stroke::new(1.0, color.linear_multiply(0.8)),
            );
        }

        // Filling with gradient
        painter.circle_filled(
            center,
            radius,
            effective_color.linear_multiply(0.9), // Slightly darker for 3D effect
        );
    }

    // surface_to_surface_connection is responsible for creating the connections between nodes from surface to surface
    fn surface_to_surface_connection(
        &self,
        painter: &Painter,
        from: Pos2,
        to: Pos2,
        from_radius: f32,
        to_radius: f32,
        color: Color32,
    ) {
        let dir = (to - from).normalized();
        let from_surface = from + dir * from_radius;
        let to_surface = to - dir * to_radius;

        painter.line_segment([from_surface, to_surface], Stroke::new(1.0, color));
    }

    fn add_inception_node(&mut self) {
        // creating the first node/index representing the world's inception
        let inception_node = Node {
            id: self.next_id,                         // id = 0
            position: na::Point3::new(0.0, 0.0, 0.0), // initial center position
            velocity: na::Vector3::zeros(),           // zero vector
            properties: NodeProperties {
                color: [1.0, 0.0, 0.0], // Red for inception node
                size: 2.0,
                strength: 1.0, // full strength
                health: 1.0,   // full health
                half_life: HALF_LIFE,
                reproduction: 0,
                generation: 0, // Root node is generation 0
                learning_factor: 0.5,
            },
            parents: None,
        };

        self.nodes.insert(self.next_id, inception_node);
        // after inserting new intial node, increment the next_id by one for the next node
        self.next_id += 1;
    }

    fn add_initial_children(&mut self) {
        // we should/are creating two initial child nodes
        for i in 0..2 {
            let position = na::Point3::new((i as f32 - 0.5) * 3.0, -3.0, 0.0);

            // Giving the child some properties
            let child = Node {
                id: self.next_id,
                position,
                velocity: na::Vector3::zeros(),
                properties: NodeProperties {
                    color: [0.0, 0.0, 1.0], // Blue for children
                    size: 1.5,
                    strength: 0.8,
                    health: 1.0,
                    half_life: HALF_LIFE,
                    reproduction: 0,
                    generation: 1, // first generation of nodes
                    learning_factor: 0.5,
                },
                parents: Some((0, 0)), // Both has inception node as parent
            };

            self.nodes.insert(self.next_id, child);
            // after inserting new nodes, we should increment the next_id by one for the next node/nodes
            self.next_id += 1;
        }
    }

    // find_clicked_node is responsible for finding the clicked node in the eco-system
    fn find_clicked_node(&self, mouse_pos: Pos2, painter: &Painter) -> Option<usize> {
        // we should convert all nodes to screen space and check distance to mouse
        for (id, node) in &self.nodes {
            let transformed_pos =
                self.view.rotation * (node.position - self.view.position.coords) * self.view.zoom;

            let screen_pos =
                painter.clip_rect().center() + vec2(transformed_pos.x, transformed_pos.y) * 100.0;

            let radius = node.properties.size * (1.0 + transformed_pos.z * 0.1) * 20.0;

            if mouse_pos.distance(screen_pos) <= radius {
                return Some(*id);
            }
        }
        None
    }

    // draw_node_hover_info in used to show the node information on mouse hover
    fn draw_node_hover_info(&self, ui: &Ui, painter: &Painter) {
        if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
            if let Some(hovered_id) = self.find_clicked_node(mouse_pos, painter) {
                if let Some(node) = self.nodes.get(&hovered_id) {
                    let transformed_pos = self.view.rotation
                        * (node.position - self.view.position.coords)
                        * self.view.zoom;
                    let screen_pos = painter.clip_rect().center()
                        + vec2(transformed_pos.x, transformed_pos.y) * 100.0;

                    // Drawing connection line from node to info card
                    let info_pos = screen_pos + vec2(100.0, -50.0);
                    painter.line_segment(
                        [screen_pos, info_pos],
                        Stroke::new(1.0, Color32::from_rgba_premultiplied(200, 200, 200, 180)),
                    );

                    // Drawing info card background
                    let card_rect = Rect::from_min_size(info_pos, vec2(150.0, 80.0));
                    painter.rect_filled(
                        card_rect,
                        5.0, // radius
                        Color32::from_rgba_premultiplied(0, 0, 0, 200),
                    );

                    let text = format!(
                        "Node #{}\nGen: {}\nHealth: {:.1}\nReproduction: {}/{}",
                        hovered_id,
                        node.properties.generation,
                        node.properties.health,
                        node.properties.reproduction,
                        MAX_REPRODUCING_ATTEMPTS
                    );

                    painter.text(
                        info_pos + vec2(10.0, 10.0),
                        egui::Align2::LEFT_TOP,
                        text,
                        egui::TextStyle::Small.resolve(ui.style()),
                        Color32::WHITE,
                    );
                }
            }
        }
    }

    fn update(&mut self) {
        // Storing IDs of nodes that will be removed from the system once they have double the half life
        let dead_nodes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.properties.health <= 0.0)
            .map(|(&id, _)| id)
            .collect();

        // Cleaning up connections for each dead node
        for dead_node_id in &dead_nodes {
            self.cleanup_dead_node_connections(*dead_node_id);
        }

        self.nodes.retain(|_, node| node.properties.health > 0.0);

        // Update health of all nodes with slow decay
        for node in self.nodes.values_mut() {
            node.properties.health *= 1.0 - HEALTH_DECAY_RATE * 0.016; // 0.016 is roughly one frame at 60fps
            node.properties.health = node.properties.health.max(0.0);
        }

        // Track average health for evolution
        let avg_health: f32 = self
            .nodes
            .values()
            .map(|n| n.properties.health)
            .sum::<f32>()
            / self.nodes.len() as f32;
        let time = Instant::now()
            .duration_since(self.last_reproduce_time)
            .as_secs_f32();
        self.avg_health_history.push((time, avg_health));
        if self.avg_health_history.len() > 1000 {
            // Limit history
            self.avg_health_history.remove(0);
        }

        // Check environmental conditions for reproduction
        let current_population = self.nodes.len();
        let ideal_population = 10000000;

        if current_population < ideal_population {
            // More favorable reproduction conditions
            if self.last_reproduce_time.elapsed().as_secs_f32() >= REPRODUCING_INTERVAL * 0.5 {
                self.attempt_reproducing();
            }
        } else {
            // Less favorable reproduction conditions
            if self.last_reproduce_time.elapsed().as_secs_f32() >= REPRODUCING_INTERVAL * 1.5 {
                self.attempt_reproducing();
            }
        }

        // Check if it's time to breed
        if self.last_reproduce_time.elapsed().as_secs_f32() >= REPRODUCING_INTERVAL {
            // Select random parents
            let node_ids: Vec<usize> = self.nodes.keys().copied().collect();
            if node_ids.len() >= 2 {
                let mut rng = rng();
                let parent1_idx = rng.random_range(0..node_ids.len());
                let parent2_idx = rng.random_range(0..node_ids.len());
                let parent1 = node_ids[parent1_idx];
                let parent2 = node_ids[parent2_idx];
                if parent1 != parent2 {
                    self.reproduce_nodes(parent1, parent2);
                }
            }
            self.last_reproduce_time = Instant::now();
        }

        // Update node positions with separation forces
        let positions: Vec<(usize, na::Point3<f32>)> = self
            .nodes
            .iter()
            .map(|(&id, node)| (id, node.position))
            .collect();

        for (id1, pos1) in positions.iter() {
            let mut separation_force = na::Vector3::zeros();

            for (id2, pos2) in positions.iter() {
                if id1 != id2 {
                    let diff = pos1 - pos2;
                    let diff_vec = na::Vector3::new(diff.x, diff.y, diff.z);
                    let distance = (diff_vec.x * diff_vec.x
                        + diff_vec.y * diff_vec.y
                        + diff_vec.z * diff_vec.z)
                        .sqrt();
                    if distance < 3.0 {
                        let normalized = diff_vec / distance;
                        separation_force +=
                            normalized * NODE_SEPARATION_FORCE * (1.0 - distance / 3.0);
                    }
                }
            }

            if let Some(node) = self.nodes.get_mut(id1) {
                node.velocity += separation_force * 0.016; // Multiply by frame time
            }
        }

        for node in self.nodes.values_mut() {
            let time = std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f32();

            let noise = OpenSimplex::new(42);
            let noise_x = noise.get([node.position.x as f64, time as f64]) as f32;
            let noise_y = noise.get([node.position.y as f64, time as f64]) as f32;
            let noise_z = noise.get([node.position.z as f64, time as f64]) as f32;

            node.velocity += na::Vector3::new(noise_x, noise_y, noise_z) * 0.01;
            node.velocity *= 0.98;
            node.position += node.velocity;
        }
    }

    /// Nodes are sorted by a quality metric (70% health, 30% strength).
    /// High-quality nodes pair within their tier, low-quality within theirs, and average nodes use learning_factor to bias toward similar quality (with some randomness).
    /// learning_factor adjusts based on child health, introducing a learning sense.
    fn attempt_reproducing(&mut self) {
        if self.last_reproduce_time.elapsed().as_secs_f32() < REPRODUCING_INTERVAL {
            return;
        }

        let node_ids: Vec<usize> = self.nodes.keys().filter(|&&id| id != 0).copied().collect();
        if node_ids.len() < 2 {
            return;
        }

        // Step 1: Compute quality scores
        let mut nodes_with_quality: Vec<(usize, f32)> = node_ids
            .iter()
            .map(|&id| {
                let node = self.nodes.get(&id).unwrap();
                let quality = 0.7 * node.properties.health + 0.3 * node.properties.strength;
                (id, quality)
            })
            .collect();
        nodes_with_quality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Step 2: Determine tiers
        let count = nodes_with_quality.len();
        let high_cutoff = count / 4;
        let low_cutoff = count - count / 4;

        // Step 3: Collect pairs to reproduce
        let mut rng = rand::rng();
        let mut pairs_to_reproduce = Vec::new();

        for i in 0..count {
            let (id1, q1) = nodes_with_quality[i];
            let node1 = self.nodes.get(&id1).unwrap();
            if node1.properties.reproduction >= MAX_REPRODUCING_ATTEMPTS {
                continue;
            }

            let mut partner_id = None;
            if i < high_cutoff {
                let candidates = &nodes_with_quality[0..high_cutoff];
                partner_id = candidates
                    .iter()
                    .filter(|&&(id, _)| id != id1)
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|&(id, _)| id);
            } else if i >= low_cutoff {
                let candidates = &nodes_with_quality[low_cutoff..];
                partner_id = candidates
                    .iter()
                    .filter(|&&(id, _)| id != id1)
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|&(id, _)| id);
            } else {
                let lf = node1.properties.learning_factor;
                let candidates = &nodes_with_quality[high_cutoff..low_cutoff];
                partner_id = candidates
                    .iter()
                    .filter(|&&(id, _)| id != id1)
                    .max_by(|a, b| {
                        let diff_a = (a.1 - q1).abs();
                        let diff_b = (b.1 - q1).abs();
                        let score_a = if rng.random::<f32>() < lf {
                            diff_a
                        } else {
                            -diff_a
                        };
                        let score_b = if rng.random::<f32>() < lf {
                            diff_b
                        } else {
                            -diff_b
                        };
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .map(|&(id, _)| id);
            }

            if let Some(id2) = partner_id {
                pairs_to_reproduce.push((id1, id2));
            }
        }

        // Step 4: Process reproduction and learning
        for (id1, id2) in pairs_to_reproduce {
            if let Some(child_id) = self.reproduce_nodes(id1, id2) {
                let child_health = self.nodes.get(&child_id).unwrap().properties.health;

                {
                    let parent1 = self.nodes.get_mut(&id1).unwrap();
                    if child_health > 0.5 {
                        parent1.properties.learning_factor =
                            (parent1.properties.learning_factor + 0.1).min(1.0);
                    } else {
                        parent1.properties.learning_factor =
                            (parent1.properties.learning_factor - 0.1).max(0.0);
                    }
                }

                {
                    let parent2 = self.nodes.get_mut(&id2).unwrap();
                    if child_health > 0.5 {
                        parent2.properties.learning_factor =
                            (parent2.properties.learning_factor + 0.1).min(1.0);
                    } else {
                        parent2.properties.learning_factor =
                            (parent2.properties.learning_factor - 0.1).max(0.0);
                    }
                }
            }
        }

        self.last_reproduce_time = Instant::now();
    }

    fn cleanup_dead_node_connections(&mut self, dead_node_id: usize) {
        // Remove references to dead nodes from other nodes' parent connections
        let nodes_to_update: Vec<usize> = self
            .nodes
            .iter()
            .filter(|(_, node)| {
                if let Some((p1, p2)) = node.parents {
                    p1 == dead_node_id || p2 == dead_node_id
                } else {
                    false
                }
            })
            .map(|(&id, _)| id)
            .collect();

        // Update the affected nodes
        for node_id in nodes_to_update {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.properties.health *= 0.5; // Reduce health when dies
                                               // Optionally remove the parent connection
                node.parents = None;
            }
        }
    }

    fn reproduce_nodes(&mut self, parent1_id: usize, parent2_id: usize) -> Option<usize> {
        // we first have to get both parent properties
        let p1_props = {
            let parent1 = self.nodes.get_mut(&parent1_id)?;
            parent1.properties.reproduction += 1;
            parent1.properties.health *= 0.95;
            parent1.properties.clone()
        };

        let p2_props = {
            let parent2 = self.nodes.get_mut(&parent2_id)?;
            parent2.properties.reproduction += 1;
            parent2.properties.health *= 0.95;
            parent2.properties.clone()
        };

        // Now we need to get their posistions on the eco-system
        let parent1_pos = self.nodes.get(&parent1_id)?.position;
        let parent2_pos = self.nodes.get(&parent2_id)?.position;

        // TODO: can different generations create a child node?
        let generation = std::cmp::max(p1_props.generation, p2_props.generation) + 1;

        let avg_health = (p1_props.health + p2_props.health) / 2.0;
        let healt_variation = rand::random::<f32>() * 0.4 - 0.2;
        let child_health = (avg_health + healt_variation).clamp(0.1, 1.0);

        // Creating child properties based on the parent properties
        let child_properties = NodeProperties {
            color: [
                (p1_props.color[0] + p2_props.color[0]) / 2.0,
                (p1_props.color[1] + p2_props.color[1]) / 2.0,
                (p1_props.color[2] + p2_props.color[2]) / 2.0,
            ],
            size: (p1_props.size + p2_props.size) / 2.0,
            strength: (p1_props.strength + p2_props.strength) / 2.0,
            health: child_health,
            half_life: HALF_LIFE,
            reproduction: 0,
            generation,
            learning_factor: 0.5,
        };

        // Calculate position between parents with slight offset
        let mid_position = (parent1_pos + parent2_pos.coords) / 2.0;

        // Create a random 3D sphere offset
        let random_angle = random::<f32>() * std::f32::consts::PI * 2.0;
        let random_z_angle = random::<f32>() * std::f32::consts::PI;
        let radius = 0.5; // Adjust this value to control spread

        let offset = na::Vector3::new(
            radius * random_angle.cos() * random_z_angle.sin(),
            radius * random_angle.sin() * random_z_angle.sin(),
            radius * random_z_angle.cos(),
        );

        let child = Node {
            id: self.next_id,
            position: mid_position + offset,
            velocity: na::Vector3::zeros(),
            properties: child_properties,
            parents: Some((parent1_id, parent2_id)),
        };

        self.nodes.insert(self.next_id, child);
        self.next_id += 1;

        Some(self.next_id - 1)
    }

    // search for a node using HashMap
    fn search_node_using_hashmap(&self, target_id: usize) -> (Option<&Node>, SearchMetrics) {
        let start_time = Instant::now();
        let mut comparisons = 0;

        comparisons += 1; // count the HashMap lookup as one comparison
        let result = self.nodes.get(&target_id);

        (
            result,
            SearchMetrics {
                algorithm: HASHMAP_DIRECT_LOOKUP.to_string(),
                comparisons,
                duration_ns: start_time.elapsed().as_nanos(),
                big_o: "O(1)".to_string(),
                space_complexity: "O(1)".to_string(),
            },
        )
    }

    // Search for a node using Linear search
    fn search_node_using_linear(&self, target_id: usize) -> (Option<&Node>, SearchMetrics) {
        let start_time = Instant::now();
        let mut comparisons = 0;

        // Linear Search Implementation
        let result = self
            .nodes
            .iter()
            .find(|(&id, _)| {
                comparisons += 1;
                id == target_id
            })
            .map(|(_, node)| node);

        let duration = start_time.elapsed();

        (
            result,
            SearchMetrics {
                algorithm: LINEAR_SEARCH.to_string(),
                comparisons,
                duration_ns: duration.as_nanos(),
                big_o: "O(n)".to_string(),
                space_complexity: "O(1)".to_string(),
            },
        )
    }

    fn search_node_using_bfs(&self, target_id: usize) -> (Option<&Node>, SearchMetrics) {
        let start_time = Instant::now();
        let mut comparisons = 0;

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        // starting from the root node (id: 0)
        if let Some(root) = self.nodes.get(&0) {
            queue.push_back(root);
            visited.insert(0);
        }

        let mut result = None;
        while let Some(node) = queue.pop_front() {
            comparisons += 1;
            if node.id == target_id {
                result = Some(node);
                break;
            }

            // adding children to queue
            for (_, child) in self.nodes.iter() {
                if let Some((p1, p2)) = child.parents {
                    if (p1 == node.id || p2 == node.id) && !visited.contains(&child.id) {
                        queue.push_back(child);
                        visited.insert(child.id);
                    }
                }
            }
        }

        (
            result,
            SearchMetrics {
                algorithm: BFS.to_string(),
                comparisons,
                duration_ns: start_time.elapsed().as_nanos(),
                big_o: "O(V + E)".to_string(),
                space_complexity: "O(V)".to_string(),
            },
        )
    }

    fn search_node_using_dsp(
        &mut self,
        start: usize,
        end: usize,
    ) -> (Option<&Node>, SearchMetrics) {
        let start_time = Instant::now();
        let mut comparisons = 0;

        let mut dist: HashMap<usize, i32> = HashMap::new();
        let mut prev: HashMap<usize, usize> = HashMap::new();
        let mut heap = BinaryHeap::new();

        // inititalising distances
        for &node_id in self.nodes.keys() {
            dist.insert(node_id, i32::max_value());
        }
        *dist.get_mut(&start).unwrap() = 0;
        heap.push(State {
            cost: 0,
            node: start,
        });

        while let Some(State { cost, node }) = heap.pop() {
            comparisons += 1;

            if node == end {
                let mut path = vec![end];
                let mut current = end;

                while let Some(&previous) = prev.get(&current) {
                    path.push(previous);
                    current = previous;
                }

                path.reverse(); // path from start to end
                self.highlighted_path = Some((path, Color32::from_rgb(255, 215, 0))); // Gold color

                return (
                    self.nodes.get(&end),
                    SearchMetrics {
                        algorithm: DSP.to_string(),
                        comparisons,
                        duration_ns: start_time.elapsed().as_nanos(),
                        big_o: "O((V + E) * log V)".to_string(),
                        space_complexity: "O(V)".to_string(),
                    },
                );
            }

            // checking if we have found a better path
            if cost > dist[&node] {
                continue;
            }

            // checking all neighbors connected through parent relationships
            for (neighbor_id, neighbor_node) in &self.nodes {
                if let Some((p1, p2)) = neighbor_node.parents {
                    if p1 == node || p2 == node {
                        let next = State {
                            cost: cost + 1,
                            node: *neighbor_id,
                        };

                        if next.cost < dist[neighbor_id] {
                            heap.push(next);
                            dist.insert(*neighbor_id, next.cost);
                            prev.insert(*neighbor_id, node);
                        }
                    }
                }
            }
        }
        (
            None,
            SearchMetrics {
                algorithm: DSP.to_string(),
                comparisons,
                duration_ns: start_time.elapsed().as_nanos(),
                big_o: "O((V + E) * log V)".to_string(),
                space_complexity: "O(V)".to_string(),
            },
        )
    }
}

impl View {
    // default returns the default configurations for the View
    fn default() -> Self {
        Self {
            position: na::Point3::new(0.0, 0.0, -10.0),
            rotation: na::UnitQuaternion::identity(),
            zoom: 1.0,
        }
    }

    // update is used to update the entire view based on the available commands.
    // This is going to implement a mixture of mouse gestures and keyboard clicks to
    // move and control the view around
    fn update(&mut self, ui: &mut Ui) {
        // Key R on the Keyboard defaults the view back to the original properties
        if ui.input(|i| i.key_pressed(Key::R)) {
            *self = View::default();
        }

        // ArrowLeft key rotates the view to the left
        if ui.input(|i| i.key_down(Key::ArrowLeft)) {
            let rotation = na::UnitQuaternion::from_euler_angles(0.0, -0.02, 0.0);
            self.rotation = rotation * self.rotation;
        }

        // ArrowRight key rotates the view to the right
        if ui.input(|i| i.key_down(Key::ArrowRight)) {
            let rotation = na::UnitQuaternion::from_euler_angles(0.0, 0.02, 0.0);
            self.rotation = rotation * self.rotation;
        }

        // ArrowUp key rotates the view upwards
        if ui.input(|i| i.key_down(Key::ArrowUp)) {
            let rotation = na::UnitQuaternion::from_euler_angles(-0.02, 0.0, 0.0);
            self.rotation = rotation * self.rotation;
        }

        // ArrowDown key rotates the view to the downwards
        if ui.input(|i| i.key_down(Key::ArrowDown)) {
            let rotation = na::UnitQuaternion::from_euler_angles(0.02, 0.0, 0.0);
            self.rotation = rotation * self.rotation;
        }

        if ui.input(|i| i.raw_scroll_delta.y) != 0.0 {
            self.zoom *= 1.0 + ui.input(|i| i.raw_scroll_delta.y) * 0.001;
            self.zoom = self.zoom.clamp(0.1, 10.0);
        }

        // Handle rotation with mouse drag
        if ui.input(|i| i.pointer.button_down(PointerButton::Secondary)) {
            let delta = ui.input(|i| i.pointer.delta());
            let rotation =
                na::UnitQuaternion::from_euler_angles(delta.y * 0.01, delta.x * 0.01, 0.0);
            self.rotation = rotation * self.rotation;
        }
    }
}

impl eframe::App for SynapticWeb {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        self.update();

        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Node Statistics");
            ui.label(format!("Total Population: {}", self.nodes.len()));

            let avg_health: f32 = self
                .nodes
                .values()
                .map(|n| n.properties.health)
                .sum::<f32>()
                / self.nodes.len() as f32;
            ui.label(format!("Average Health: {:.2}", avg_health));

            ui.label(format!(
                "Last Reproduction Time: {:#?}",
                self.last_reproduce_time
            ));

            ui.separator();
            ui.heading("Controls");
            ui.label("Mouse:");
            ui.label("• Right Click + Drag: Rotate Camera");
            ui.label("• Scroll: Zoom");
            ui.label("• Left Click: Select Node");

            ui.separator();
            ui.label("Keyboard:");
            ui.label("Arrow keys: Rotate the view respectively");
            ui.label("• R: Reset Camera");
            ui.label("• Space: Center on Root Node");
            ui.label("• C: Clear Selection");
            ui.label("• ESC: Exit");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.view.update(ui);
            let painter = ui.painter();

            self.draw_node_hover_info(ui, &painter);

            if ui.input(|i| i.key_pressed(Key::Space)) {
                // center on root node (id: 0)
                if let Some(root) = self.nodes.get(&0) {
                    self.view.position = root.position;
                }
            }

            if ui.input(|i| i.key_pressed(Key::C)) {
                self.selected_node = None;
            };

            // mouse interaction for node selection
            if ui.input(|i| i.pointer.primary_clicked()) {
                let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                if let Some(mouse_pos) = mouse_pos {
                    self.selected_node = self.find_clicked_node(mouse_pos, &painter);
                }
            }

            // First pass: Draw connections
            // First pass: Draw connections
            for node in self.nodes.values() {
                if let Some((parent1_id, parent2_id)) = node.parents {
                    if let (Some(parent1), Some(parent2)) =
                        (self.nodes.get(&parent1_id), self.nodes.get(&parent2_id))
                    {
                        let child_pos = self.view.rotation
                            * (node.position - self.view.position.coords)
                            * self.view.zoom;
                        let parent1_pos = self.view.rotation
                            * (parent1.position - self.view.position.coords)
                            * self.view.zoom;
                        let parent2_pos = self.view.rotation
                            * (parent2.position - self.view.position.coords)
                            * self.view.zoom;

                        let screen_child =
                            painter.clip_rect().center() + vec2(child_pos.x, child_pos.y) * 100.0;
                        let screen_parent1 = painter.clip_rect().center()
                            + vec2(parent1_pos.x, parent1_pos.y) * 100.0;
                        let screen_parent2 = painter.clip_rect().center()
                            + vec2(parent2_pos.x, parent2_pos.y) * 100.0;

                        let connection_color =
                            if let Some((ref path, color)) = self.highlighted_path {
                                // Check if this connection is in the path (parent1 -> node or parent2 -> node)
                                let in_path = path.windows(2).any(|w| {
                                    (w[0] == parent1_id && w[1] == node.id)
                                        || (w[0] == parent2_id && w[1] == node.id)
                                });
                                if in_path {
                                    color
                                } else {
                                    Color32::GRAY
                                }
                            } else {
                                Color32::GRAY
                            };

                        self.surface_to_surface_connection(
                            &painter,
                            screen_child,
                            screen_parent1,
                            node.properties.size * 20.0,
                            parent1.properties.size * 20.0,
                            connection_color,
                        );
                        self.surface_to_surface_connection(
                            &painter,
                            screen_child,
                            screen_parent2,
                            node.properties.size * 20.0,
                            parent2.properties.size * 20.0,
                            connection_color,
                        );
                    }
                }
            }

            // Second pass: Draw nodes as 3D spheres
            for node in self.nodes.values() {
                let transformed_pos = self.view.rotation
                    * (node.position - self.view.position.coords)
                    * self.view.zoom;

                let screen_pos = painter.clip_rect().center()
                    + vec2(transformed_pos.x, transformed_pos.y) * 100.0;

                let radius = node.properties.size * (1.0 + transformed_pos.z * 0.1) * 20.0;

                self.draw_3d_node(
                    &painter,
                    screen_pos,
                    radius,
                    Color32::from_rgb(
                        (node.properties.color[0] * 255.0) as u8,
                        (node.properties.color[1] * 255.0) as u8,
                        (node.properties.color[2] * 255.0) as u8,
                    ),
                );
            }

            // Display node info window when a node is selected
            if let Some(selected_id) = self.selected_node {
                if let Some(node) = self.nodes.get(&selected_id) {
                    egui::Window::new(format!("Node #{}", selected_id)).show(ctx, |ui| {
                        ui.label(format!("Generation: {}", node.properties.generation));
                        ui.label(format!("Health: {:.2}", node.properties.health));
                        ui.label(format!("Size: {:.2}", node.properties.size));
                        ui.label(format!("Strength: {:.2}", node.properties.strength));
                        ui.label(format!(
                            "Breeding Attempts: {}/{}",
                            node.properties.reproduction, MAX_REPRODUCING_ATTEMPTS
                        ));

                        if let Some((p1, p2)) = node.parents {
                            ui.label(format!("Parents: #{}, #{}", p1, p2));
                        } else {
                            ui.label("Root Node");
                        }
                    });
                }
            }
        });

        egui::Window::new("Network Analysis").show(ctx, |ui| {
            let avg_degree = self.nodes.values().filter_map(|n| n.parents).count() as f32 * 2.0
                / self.nodes.len() as f32;
            ui.label(format!("Average Degree: {:.2}", avg_degree));
            if ui.button("Find Central Node").clicked() {
                let central_id = self
                    .nodes
                    .iter()
                    .max_by_key(|(_, n)| {
                        self.nodes
                            .iter()
                            .filter(|(_, m)| {
                                m.parents.map_or(false, |(p1, p2)| p1 == n.id || p2 == n.id)
                            })
                            .count()
                    })
                    .map(|(&id, _)| id);
                self.selected_node = central_id;
            }
        });

        egui::Window::new("Search Node").show(ctx, |ui| {
            // let mut search_id: String = String::new();
            // let mut selected_algo: SearchAlgorithm = SearchAlgorithm::HashMap;
            // let mut search_results: Vec<SearchMetrics> = Vec::new();

            ui.horizontal(|ui| {
                ui.label("Search Algorithm:");
                ui.radio_value(&mut self.selected_algo, SearchAlgorithm::HashMap, "HashMap");
                ui.radio_value(&mut self.selected_algo, SearchAlgorithm::Linear, "Linear");
                ui.radio_value(&mut self.selected_algo, SearchAlgorithm::BFS, "BFS");
                ui.radio_value(&mut self.selected_algo, SearchAlgorithm::DSP, "DSP");
                ui.radio_value(
                    &mut self.selected_algo,
                    SearchAlgorithm::CompareAll,
                    "Compare All",
                );
            });

            ui.horizontal(|ui| {
                let _text_edit = ui.text_edit_singleline(&mut self.search_id);
                if ui.button("Search").clicked() {
                    if let Ok(id) = self.search_id.parse::<usize>() {
                        self.search_results.clear();
                        match self.selected_algo {
                            SearchAlgorithm::HashMap => {
                                let (_, metrics) = self.search_node_using_hashmap(id);
                                self.search_results.push(metrics);
                            }
                            SearchAlgorithm::Linear => {
                                let (_, metrics) = self.search_node_using_linear(id);
                                self.search_results.push(metrics);
                            }
                            SearchAlgorithm::BFS => {
                                let (_, metrics) = self.search_node_using_bfs(id);
                                self.search_results.push(metrics);
                            }
                            SearchAlgorithm::DSP => {
                                let (_, metrics) = self.search_node_using_dsp(0, id);
                                self.search_results.push(metrics);
                            }
                            SearchAlgorithm::CompareAll => {
                                let (_, m1) = self.search_node_using_hashmap(id);
                                let (_, m2) = self.search_node_using_linear(id);
                                let (_, m3) = self.search_node_using_bfs(id);
                                let (_, m4) = self.search_node_using_dsp(0, id);
                                self.search_results.extend(vec![m1, m2, m3, m4]);
                            }
                        }
                        self.search_id.clear();
                    }
                }
            });

            if !self.search_results.is_empty() {
                ui.separator();
                ui.heading("Search Results");

                for metrics in &self.search_results {
                    ui.group(|ui| {
                        ui.label(format!("Algorithm: {}", metrics.algorithm));
                        ui.label(format!("Comparisons: {}", metrics.comparisons));
                        ui.label(format!(
                            "Duration: {:.2}µs",
                            metrics.duration_ns as f64 / 1000.0
                        ));
                        ui.label(format!("Time Complexity: {}", metrics.big_o));
                        ui.label(format!("Space Complexity: {}", metrics.space_complexity));
                    });
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SynapticWeb",
        options,
        Box::new(|_cc| Ok(Box::new(SynapticWeb::new()))),
    )
}
