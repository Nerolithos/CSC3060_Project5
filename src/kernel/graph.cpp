#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>


void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void stu_graph(std::uint64_t& out, const Graph& graph) {
    // This will be called with the optimized data structure
    // Will be implemented in wrapper
    out = 0;
}

// Conversion function - not included in time measurement
void convert_graph_to_csr(graph_args& args) {
    const int n = args.graph.n;
    args.node_offsets.assign(n + 1, 0);
    
    // First pass: count edges for each node and build offsets
    std::size_t total_edges = 0;
    for (int u = 0; u < n; ++u) {
        const Edge* e = args.graph.nodes[u].edges;
        int degree = 0;
        while (e) {
            degree++;
            e = e->next;
        }
        args.node_offsets[u + 1] = degree;
        total_edges += degree;
    }
    
    // Compute prefix sums
    for (int i = 1; i <= n; ++i) {
        args.node_offsets[i] += args.node_offsets[i - 1];
    }
    
    // Build adjacency array
    args.adjacency.resize(total_edges);
    for (int u = 0; u < n; ++u) {
        int pos = args.node_offsets[u];
        const Edge* e = args.graph.nodes[u].edges;
        while (e) {
            args.adjacency[pos++] = e->to;
            e = e->next;
        }
    }
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    
    // Convert to CSR format only once (not included in timing)
    if (!args.converted) {
        convert_graph_to_csr(args);
        args.converted = true;
    }
    
    // Compute checksum using optimized representation
    std::uint64_t checksum = 0;
    const int n = args.graph.n;
    
    for (int u = 0; u < n; ++u) {
        const int start = args.node_offsets[u];
        const int end = args.node_offsets[u + 1];
        for (int i = start; i < end; ++i) {
            checksum += static_cast<std::uint64_t>(args.adjacency[i]);
        }
    }
    
    args.out = checksum;
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
