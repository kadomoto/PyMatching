// Copyright 2020 Oscar Higgott

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include "lemon_mwpm.h"
#include <lemon/list_graph.h>
#include <lemon/matching.h>
#include <lemon/connectivity.h>
#include <vector>
#include <string>
#include "weighted_stabiliser_graph.h"
#include "stabiliser_graph.h"
#include <stdexcept>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <boost/graph/adjacency_list.hpp>

typedef lemon::ListGraph UGraph;
typedef UGraph::EdgeMap<double> LengthMap;
typedef lemon::MaxWeightedPerfectMatching<UGraph,LengthMap> MWPM;


const char * BlossomFailureException::what() const throw() {
    return "The Lemon implementation of the blossom algorithm "
            "(lemon::MaxWeightedPerfectMatching) "
            "was unable to find a solution to the minimum-weight "
            "perfect matching problem.";
}


class DefectGraph {
    public:
        DefectGraph(int num_nodes);
        void AddEdge(int i, int j, double weight);
        UGraph g;
        LengthMap length;
        int num_nodes;
};

DefectGraph::DefectGraph(int num_nodes) : num_nodes(num_nodes),
         length(g)
{
    for (int i=0; i<num_nodes; i++){
        UGraph::Node x;
        x = g.addNode();
    }
}

void DefectGraph::AddEdge(int i, int j, double weight){
    UGraph::Edge e = g.addEdge(g.nodeFromId(i), g.nodeFromId(j));
    length[e] = weight;
}


MatchingResult LemonDecode(
    WeightedStabiliserGraph& sg,
    const py::array_t<int>& defects,
    bool return_weight
    ){
    MatchingResult matching_result;
    if (!sg.HasComputedAllPairsShortestPaths()){
        sg.ComputeAllPairsShortestPaths();
    }
    int num_nodes = sg.GetNumNodes();

    auto d = defects.unchecked<1>();
    std::set<int> defects_set;
    for (int i=0; i<d.shape(0); i++){
        if (d(i) >= num_nodes){
            throw std::invalid_argument(
            "Defect id must be less than the number of nodes in the matching graph"
            );
        }
        defects_set.insert(d(i));
    }
    sg.FlipBoundaryNodesIfNeeded(defects_set);

    std::vector<int> defects_vec(defects_set.begin(), defects_set.end());

    int num_defects = defects_vec.size();

    DefectGraph defect_graph(num_defects);

    for (py::size_t i = 0; i<num_defects; i++){
        for (py::size_t j=i+1; j<num_defects; j++){
            defect_graph.AddEdge(i, j, -1.0*sg.SpaceTimeDistance(
                defects_vec[i], defects_vec[j]
                ));
        }
    };

    MWPM pm(defect_graph.g, defect_graph.length);
    bool success = pm.run();
    if (!success){
        throw BlossomFailureException();
    }

    int N = sg.GetNumQubits();
    auto correction = new std::vector<int>(N, 0);
    std::set<int> qids;
    for (py::size_t i = 0; i<num_defects; i++){
        int j = defect_graph.g.id(pm.mate(defect_graph.g.nodeFromId(i)));
        if (i<j){
            std::vector<int> path = sg.SpaceTimeShortestPath(
                defects_vec[i], defects_vec[j]
                );
            for (std::vector<int>::size_type k=0; k<path.size()-1; k++){
                qids = sg.QubitIDs(path[k], path[k+1]);
                for (auto qid : qids){
                    if ((qid != -1) && (qid >= 0) && (qid < N)){
                        (*correction)[qid] = ((*correction)[qid] + 1) % 2;
                    }
                }
            }
        }
    }

    auto capsule = py::capsule(correction, [](void *correction) { delete reinterpret_cast<std::vector<int>*>(correction); });
    auto corr = py::array_t<int>(correction->size(), correction->data(), capsule);

    if (return_weight) {
        matching_result.weight = -1*pm.matchingWeight();
    } else {
        matching_result.weight = -1.0;
    }

    matching_result.correction = corr;
    return matching_result;
}


MatchingResult LemonDecodeMatchNeighbourhood(WeightedStabiliserGraph& sg, const py::array_t<int>& defects, 
                                             int num_neighbours, bool return_weight){
    MatchingResult matching_result;
    auto d = defects.unchecked<1>();

    int num_nodes = sg.GetNumNodes();

    std::set<int> defects_set;
    for (int i=0; i<d.shape(0); i++) {
        if (d(i) >= num_nodes){
            throw std::invalid_argument(
            "Defect id must be less than the number of nodes in the matching graph"
            );
        }
        defects_set.insert(d(i));
    }

    sg.FlipBoundaryNodesIfNeeded(defects_set);

    std::vector<int> defects_vec(defects_set.begin(), defects_set.end());
    int num_defects = defects_vec.size();
    std::vector<int> defect_id(num_nodes, -1);
    for (int i=0; i<num_defects; i++){
        defect_id[defects_vec[i]] = i;
    }
    num_neighbours = std::min(num_neighbours, num_defects-1) + 1;

    DefectGraph defect_graph(num_defects);

    std::vector<std::pair<int, double>> neighbours;
    int j;
    bool is_in;
    for (int i=0; i<num_defects; i++){
        neighbours = sg.GetNearestNeighbours(defects_vec[i], num_neighbours, defect_id);
        for (const auto &neighbour : neighbours){
            j = defect_id[neighbour.first];
            UGraph::Edge FoundEdge = lemon::findEdge(
                defect_graph.g,
                defect_graph.g.nodeFromId(i),
                defect_graph.g.nodeFromId(j));
            is_in = FoundEdge != lemon::INVALID;
            if (!is_in && i!=j){
                defect_graph.AddEdge(i, j, -1.0*neighbour.second);
            }
        }
    }

    MWPM pm(defect_graph.g, defect_graph.length);
    bool success = pm.run();
    if (!success){
        throw BlossomFailureException();
    }

    int N = sg.GetNumQubits();
    auto correction = new std::vector<int>(N, 0);

    std::set<int> remaining_defects;
    for (int i=0; i<num_defects; i++){
        remaining_defects.insert(i);
    }

    std::vector<int> path;
    int i;
    std::set<int> qids;
    while (remaining_defects.size() > 0){
        i = *remaining_defects.begin();
        remaining_defects.erase(remaining_defects.begin());
        j = defect_graph.g.id(pm.mate(defect_graph.g.nodeFromId(i)));
        remaining_defects.erase(j);
        path = sg.GetPath(defects_vec[i], defects_vec[j]);
        for (std::vector<int>::size_type k=0; k<path.size()-1; k++){
            qids = sg.QubitIDs(path[k], path[k+1]);
            for (auto qid : qids){
                if ((qid != -1) && (qid >= 0) && (qid < N)){
                    (*correction)[qid] = ((*correction)[qid] + 1) % 2;
                }
            }
        }
    }
    auto capsule = py::capsule(correction, [](void *correction) { delete reinterpret_cast<std::vector<int>*>(correction); });
    auto corr = py::array_t<int>(correction->size(), correction->data(), capsule);

    if (return_weight) {
        matching_result.weight = -1*pm.matchingWeight();
    } else {
        matching_result.weight = -1.0;
    }
    
    matching_result.correction = corr;
    return matching_result;
}