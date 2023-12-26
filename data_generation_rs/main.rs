use ndarray::{Array2, arr2};
use std::rc::Rc;
use itertools::Itertools;

struct Tree {
    n_tasks: usize,
    m_machines: usize,
    working_time_matrix: Array2<f64>,
    root: Node,
}

impl Tree {
    fn new(working_time_matrix: Array2<f64>) -> Self {
        let (n_tasks, m_machines) = working_time_matrix.dim();
        let root = Node::new(None, Vec::new(), m_machines, working_time_matrix.clone());
        Tree {
            n_tasks,
            m_machines,
            working_time_matrix,
            root,
        }
    }

    fn branch_and_bound(&mut self) -> Option<Rc<Node>> {
        fn branch_and_bound_cut(
            tree: &Tree,
            node_value: f64,
            node: &Node,
            n_tasks: usize,
            ub: f64,
        ) -> bool {
            let not_used_machines: Vec<usize> = (0..n_tasks)
                .filter(|task| !node.tasks.contains(task))
                .collect();
            let minimal_value = node_value
                + tree
                    .working_time_matrix
                    .slice(s![not_used_machines, tree.m_machines - 1])
                    .sum::<f64>();
            minimal_value > ub
        }

        self.beam_search(&branch_and_bound_cut)
    }

    fn beam_search(
        &mut self,
        cut: &dyn Fn(&Tree, f64, &Node, usize, f64) -> bool,
    ) -> Option<Rc<Node>> {
        let (best_node, _) = self.get_best(&self.root, self.n_tasks, cut, f64::INFINITY);
        best_node
    }

    fn get_best(
        &self,
        node: &Node,
        n_tasks: usize,
        cut: &dyn Fn(&Tree, f64, &Node, usize, f64) -> bool,
        mut ub: f64,
    ) -> (Option<Rc<Node>>, f64) {
        let mut node_value = node.value();
        if cut(self, node_value, node, n_tasks, ub) {
            return (None, ub);
        }
        if node.tasks.len() == n_tasks {
            return (Some(Rc::new(node.clone())), node_value);
        }
        let mut best_node = None;
        for task in (0..n_tasks).filter(|t| !node.tasks.contains(t)) {
            let new_node = Node::new(
                Some(Rc::clone(node)),
                [&node.tasks[..], &[task]].concat(),
                self.m_machines,
                self.working_time_matrix.clone(),
            );
            let (new_node, new_ub) = self.get_best(&new_node, n_tasks, cut, ub);
            if new_ub < ub {
                ub = new_ub;
                best_node = new_node;
            }
        }
        (best_node, ub)
    }
}

struct Node {
    parent: Option<Rc<Node>>,
    tasks: Vec<i32>,
    m_machines: usize,
    working_time_matrix: Array2<f64>,
    children: Vec<Rc<Node>>,
    state: Option<Array2<f64>>,
    value: f64,
}

impl Node {
    fn new(
        parent: Option<Rc<Node>>,
        tasks: Vec<i32>,
        m_machines: usize,
        working_time_matrix: Array2<f64>,
        children: Vec<Rc<Node>>,
        state: Option<Array2<f64>>,
        value: f64,
    ) -> Self {
        Node {
            parent,
            tasks,
            m_machines,
            working_time_matrix,
            children,
            state,
            value,
        }
    }

    fn fill_state(&mut self) {
        if let Some(last_task) = self.tasks.last() {
            let mut new_state = Array2::<f64>::zeros((1, self.m_machines));
            let working_time = &self.working_time_matrix.slice(s![*last_task, ..]);
            if let Some(ref parent_state) = self.state {
                let mut prev_time = 0.0;
                for ((work_time, total_time_at_previous_task), state_value) in working_time
                    .into_iter()
                    .zip(parent_state.slice(s![0, ..]).iter())
                {
                    let current_time = work_time + prev_time.max(*state_value);
                    new_state[[0, *work_time as usize]] = current_time;
                    prev_time = current_time;
                }
            } else {
                new_state = working_time.clone().insert_axis(Axis(0));
            }
            self.state = Some(new_state);
        }
    }

    fn get_state(&mut self) -> Option<Array2<f64>> {
        match self.state {
            Some(ref state) => Some(state.clone()),
            None => {
                if self.tasks.is_empty() {
                    Some(arr2(&[[]]))
                } else {
                    if let Some(ref parent) = self.parent {
                        if let Some(ref parent_state) = parent.get_state() {
                            let mut new_state = Array2::<f64>::zeros((1, self.m_machines));
                            new_state.slice_mut(s![0, ..]).assign(parent_state);
                            new_state = new_state.insert_axis(Axis(0));
                            self.state = Some(new_state);
                            self.fill_state();
                        }
                    } else {
                        let last_task = self.tasks.last().unwrap();
                        let new_state = self.working_time_matrix.slice(s![*last_task, ..])
                            .insert_axis(Axis(0));
                        self.state = Some(new_state);
                    }
                    self.state.clone()
                }
            }
        }
    }
}
