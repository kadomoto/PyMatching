#include "mwpm.h"


void pm::Mwpm::add_detection_event(int detector_node_id) {
    auto node = &flooder.graph.nodes[detector_node_id];
    flooder.create_region(node);
    detection_events.push_back(node);
}

pm::Mwpm::Mwpm(pm::GraphFlooder& flooder) : flooder(std::move(flooder)) {}


void pm::Mwpm::shatter_descendants_into_matches_and_freeze(pm::AltTreeNode &alt_tree_node) {
    if (alt_tree_node.inner_region) {
        alt_tree_node.parent = pm::AltTreeEdge();
        alt_tree_node.inner_region->add_match(
                alt_tree_node.outer_region, alt_tree_node.inner_to_outer_edge
        );
        flooder.set_region_frozen(*alt_tree_node.inner_region);
        flooder.set_region_frozen(*alt_tree_node.outer_region);
        alt_tree_node.inner_region->alt_tree_node = nullptr;
        alt_tree_node.outer_region->alt_tree_node = nullptr;
    }
    for (auto& child_edge : alt_tree_node.children) {
        shatter_descendants_into_matches_and_freeze(*child_edge.alt_tree_node);
    }
    delete &alt_tree_node;
}

void pm::Mwpm::handle_tree_hitting_boundary(const pm::RegionHitBoundaryEventData &event) {
    auto node = event.region->alt_tree_node;
    node->become_root();
    // Match descendents, deleting AltTreeNodes and freezing GraphFillRegions
    shatter_descendants_into_matches_and_freeze(*node);

    // Now match the event region to the boundary and freeze
    event.region->match = pm::Match(
            nullptr, event.edge
    );
    flooder.set_region_frozen(*event.region);
}

void pm::Mwpm::handle_tree_hitting_boundary_match(
        pm::GraphFillRegion *unmatched_region,
        pm::GraphFillRegion *matched_region,
        const pm::CompressedEdge &unmatched_to_matched_edge
        ) {
    auto& alt_tree_node = unmatched_region->alt_tree_node;
    unmatched_region->add_match(matched_region, unmatched_to_matched_edge);
    flooder.set_region_frozen(*unmatched_region);
    alt_tree_node->become_root();
    shatter_descendants_into_matches_and_freeze(*alt_tree_node);
}

void pm::Mwpm::handle_tree_hitting_other_tree(const pm::RegionHitRegionEventData &event) {
    auto alt_node_1 = event.region1->alt_tree_node;
    auto alt_node_2 = event.region2->alt_tree_node;
    // Tree rotation
    event.region1->alt_tree_node->become_root();
    event.region2->alt_tree_node->become_root();
    // Match and freeze descendants
    shatter_descendants_into_matches_and_freeze(*alt_node_1);
    shatter_descendants_into_matches_and_freeze(*alt_node_2);
    // Match colliding nodes
    event.region1->add_match(event.region2, event.edge);
    // Freeze colliding regions
    flooder.set_region_frozen(*event.region1);
    flooder.set_region_frozen(*event.region2);
}

void pm::Mwpm::handle_tree_hitting_match(pm::GraphFillRegion *unmatched_region, pm::GraphFillRegion *matched_region,
                                         const pm::CompressedEdge &unmatched_to_matched_edge) {
    auto alt_tree_node = unmatched_region->alt_tree_node;
    alt_tree_node->make_child(
            matched_region, matched_region->match.region,
            matched_region->match.edge, unmatched_to_matched_edge
            );
    auto other_match = matched_region->match.region;
    other_match->match = Match();
    matched_region->match = Match();
    flooder.set_region_shrinking(*matched_region);
    flooder.set_region_growing(*other_match);
}

void pm::Mwpm::handle_tree_hitting_self(const pm::RegionHitRegionEventData &event, pm::AltTreeNode *common_ancestor) {
    auto alt_node_1 = event.region1->alt_tree_node;
    auto alt_node_2 = event.region2->alt_tree_node;
    auto prune_result_1 = alt_node_1->prune_upward_back_edge_path_stopping_before(
            common_ancestor
            );
    auto prune_result_2 = alt_node_2->prune_upward_path_stopping_before(common_ancestor);

    // Construct blossom region cycle
    auto blossom_cycle = std::move(prune_result_2.pruned_path_region_edges);
    auto p1s = prune_result_1.pruned_path_region_edges.size();
    blossom_cycle.reserve(blossom_cycle.size() + p1s + 1);
    for (size_t i = 0; i < p1s; i++)
        blossom_cycle.push_back(prune_result_1.pruned_path_region_edges[p1s-i-1]);
    blossom_cycle.emplace_back(
            event.region1, event.edge
    );
    auto blossom_region = flooder.create_blossom(blossom_cycle);

    common_ancestor->outer_region = blossom_region;
    blossom_region->alt_tree_node = common_ancestor;
    common_ancestor->children.reserve(common_ancestor->children.size() + prune_result_1.orphan_edges.size()
                                        + prune_result_2.orphan_edges.size());
    for (auto& c : prune_result_1.orphan_edges) {
        common_ancestor->add_child(c);
    }
    for (auto& c : prune_result_2.orphan_edges) {
        common_ancestor->add_child(c);
    }
}

void pm::Mwpm::handle_blossom_shattering(const pm::BlossomShatterEventData &event) {
    // First find indices of in_parent_region and in_child_region
    // in_parent_region is the blossom cycle region connected to the parent of the blossom inner node.
    // in_child_region is the blossom cycle region connected to the child of the inner node
    auto& blossom_cycle = event.blossom_region->blossom_children;
    auto blossom_alt_node = event.blossom_region->alt_tree_node;
    size_t bsize = blossom_cycle.size();
    size_t parent_idx, child_idx;
    for (size_t i = 0; i < bsize; i++) {
        if (blossom_cycle[i].region == event.in_parent_region){
            parent_idx = i;
        } else if (blossom_cycle[i].region == event.in_child_region) {
            child_idx = i;
        }
    }

    // Length of path starting on in_parent and stopping before in_child
    size_t gap = ((child_idx - parent_idx) % bsize);
    AltTreeNode* current_alt_node;
    size_t evens_start, evens_end;

    current_alt_node = event.blossom_region->alt_tree_node->parent.alt_tree_node;
    pm::unstable_erase(current_alt_node->children, [blossom_alt_node](AltTreeEdge x){
        return x.alt_tree_node == blossom_alt_node;
    });
    auto child_edge = blossom_alt_node->parent.edge.reversed();
    size_t k1, k2;

    if (gap % 2 == 0) {
        // The path starting after in_child and stopping before in_parent is even length. Regions will
        // be matched along this path
        evens_start = child_idx + 1;
        evens_end = child_idx + bsize - gap;

        // Now insert odd-length path starting on in_parent and ending on in_child into alternating tree
        for (size_t i = parent_idx; i < parent_idx + gap; i += 2) {
            current_alt_node = current_alt_node->make_child(
                    blossom_cycle[i].region,
                    blossom_cycle[i+1].region,
                    blossom_cycle[i].edge,
                    child_edge
                    );
            child_edge = blossom_cycle[i+1].edge;
            flooder.set_region_shrinking(*current_alt_node->inner_region);
            flooder.set_region_growing(*current_alt_node->outer_region);
        }
    } else {
        // The path starting after in_parent and stopping before in_child is even length. Regions will
        // be matched along this path
        evens_start = parent_idx + 1;
        evens_end = parent_idx + gap;

        // Now insert odd-length path into alternating tree
        size_t k3;
        for (size_t i = 0; i < bsize - gap; i += 2) {
            k1 = (parent_idx + bsize - i) % bsize;
            k2 = (parent_idx + bsize - i - 1) % bsize;
            k3 = (parent_idx + bsize - i - 2) % bsize;
            current_alt_node = current_alt_node->make_child(
                    blossom_cycle[k1].region,
                    blossom_cycle[k2].region,
                    blossom_cycle[k2].edge.reversed(),
                    child_edge
            );
            child_edge = blossom_cycle[k3].edge.reversed();
            flooder.set_region_shrinking(*current_alt_node->inner_region);
            flooder.set_region_growing(*current_alt_node->outer_region);
        }

    }

    for (size_t j = evens_start; j < evens_end; j+=2) {
        k1 = j % bsize;
        k2 = (j + 1) % bsize;
        blossom_cycle[k1].region->add_match(
                blossom_cycle[k2].region,
                blossom_cycle[k1].edge
        );
        flooder.reschedule_events_for_region(*blossom_cycle[k1].region);
        flooder.reschedule_events_for_region(*blossom_cycle[k2].region);
    }

    blossom_alt_node->inner_region = blossom_cycle[child_idx].region;
    flooder.set_region_shrinking(*blossom_alt_node->inner_region);
    blossom_cycle[child_idx].region->alt_tree_node = blossom_alt_node;
    current_alt_node->add_child(AltTreeEdge(
            blossom_alt_node, child_edge
    ));

    delete event.blossom_region;
}

void pm::Mwpm::process_event(const pm::MwpmEvent &event) {
    if (event.event_type == pm::REGION_HIT_REGION) {
        auto alt_node_1 = event.region_hit_region_event_data.region1->alt_tree_node;
        auto alt_node_2 = event.region_hit_region_event_data.region2->alt_tree_node;
        if (alt_node_1 && alt_node_2) {
            auto common_ancestor = alt_node_1->most_recent_common_ancestor(*alt_node_2);
            if (!common_ancestor) {
                handle_tree_hitting_other_tree(event.region_hit_region_event_data);
            } else {
                handle_tree_hitting_self(event.region_hit_region_event_data, common_ancestor);
            }
        } else if (alt_node_1) {
            // Region 2 is not in the tree, so must be matched to the boundary or another region
            if (event.region_hit_region_event_data.region2->match.region) {
                handle_tree_hitting_match(
                        event.region_hit_region_event_data.region1,
                        event.region_hit_region_event_data.region2,
                        event.region_hit_region_event_data.edge
                );
                } else {
                handle_tree_hitting_boundary_match(
                        event.region_hit_region_event_data.region1,
                        event.region_hit_region_event_data.region2,
                        event.region_hit_region_event_data.edge
                        );
            }
        } else {
            // Region 1 is not in the tree, so must be matched to the boundary or another region
            if (event.region_hit_region_event_data.region1->match.region) {
                handle_tree_hitting_match(
                        event.region_hit_region_event_data.region2,
                        event.region_hit_region_event_data.region1,
                        event.region_hit_region_event_data.edge.reversed()
                );
            } else {
                handle_tree_hitting_boundary_match(
                        event.region_hit_region_event_data.region2,
                        event.region_hit_region_event_data.region1,
                        event.region_hit_region_event_data.edge.reversed()
                );
            }
        }
    } else if (event.event_type == pm::REGION_HIT_BOUNDARY) {
        handle_tree_hitting_boundary(event.region_hit_boundary_event_data);
    } else {
        handle_blossom_shattering(event.blossom_shatter_event_data);
    }
}

