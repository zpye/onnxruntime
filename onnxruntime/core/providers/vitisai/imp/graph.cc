// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/graph.h"

#include <codecvt>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <string>

#include "core/graph/model_saving_options.h"
#include "core/providers/shared_library/provider_api.h"
#include "./vai_assert.h"

#include "vaip/node.h"
#include "vaip/node_arg.h"
#include "./tensor_proto.h"
namespace vaip {

struct NodeEdgeT {
  const onnxruntime::NodeIndex src_node_index;
  const onnxruntime::NodeIndex dst_node_index;
  const int src_arg_index;
  const int dst_arg_index;
};

static void graph_remove_node(Graph& graph, const Node& node) {
  auto remove_edges = std::vector<NodeEdgeT>();
  for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
    remove_edges.push_back(NodeEdgeT{it->GetNode().Index(), node.Index(), it->GetSrcArgIndex(), it->GetDstArgIndex()});
  }
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    remove_edges.push_back(NodeEdgeT{node.Index(), it->GetNode().Index(), it->GetSrcArgIndex(), it->GetDstArgIndex()});
  }
  for (auto it : remove_edges) {
    graph.RemoveEdge(it.src_node_index, it.dst_node_index, it.src_arg_index, it.dst_arg_index);
  }
  graph.RemoveNode(node.Index());
}

static std::vector<const NodeArg*> node_get_implicit_input_node_args(const Node& node) {
  auto ret = std::vector<const NodeArg*>();
  auto implicit_input_defs = node.ImplicitInputDefs();
  ret.reserve(implicit_input_defs.size());
  for (auto i : implicit_input_defs) {
    ret.push_back(i);
  }
  return ret;
}
Node& graph_add_node(Graph& graph, const std::string& name, const std::string& op_type, const std::string& description,
                     const std::vector<const NodeArg*>& input_args, const std::vector<const NodeArg*>& output_args,
                     const NodeAttributes& attributes, const std::string& domain) {
  std::vector<NodeArg*> inputs;
  inputs.reserve(input_args.size());
  for (auto i : input_args) {
    inputs.push_back(const_cast<NodeArg*>(i));
  }
  std::vector<NodeArg*> outputs;
  outputs.reserve(output_args.size());
  for (auto i : output_args) {
    outputs.push_back(const_cast<NodeArg*>(i));
  }
  auto& ret = graph.AddNode(name, op_type, description, inputs, outputs, &attributes, domain);
  auto src_arg_index = 0;
  for (auto& o : outputs) {
    auto consumers = graph.GetConsumerNodes(o->Name());
    for (auto& consumer : consumers) {
      auto dst_arg_index = 0u;
      auto tmp_inputs = node_get_inputs(*consumer);
      for (auto ni : *tmp_inputs) {
        auto name1 = ni.node_arg->Name();
        if (name1 == o->Name()) {
          graph.AddEdge(ret.Index(), consumer->Index(), src_arg_index, dst_arg_index);
        }
        dst_arg_index = dst_arg_index + 1;
      }
      // dst_arg_index should not init again.
      for (auto implicit_node_arg : node_get_implicit_input_node_args(*consumer)) {
        auto name1 = implicit_node_arg->Name();
        if (name1 == o->Name()) {
          graph.AddEdge(ret.Index(), consumer->Index(), src_arg_index, dst_arg_index);
        }
        dst_arg_index = dst_arg_index + 1;
      }
    }
    src_arg_index = src_arg_index + 1;
  }
  return ret;
}

// copied from graph.cc, trying to exit the function early as leave function may change the validity of the graph
void graph_reverse_dfs_from(
    const Graph& graph, gsl::span<const Node* const> from,
    const std::function<bool(const Node*)>& enter,
    const std::function<bool(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& comp,
    const std::function<bool(const Node* from, const Node* to)>&
        stop) {
  using WorkEntry = std::pair<const Node*, bool>;  // bool represents leave or not
  InlinedVector<WorkEntry> stack;
  stack.reserve(from.size());
  for (auto node : from) {
    stack.emplace_back(node, false);
  }

  InlinedVector<bool> visited(graph.MaxNodeIndex(), false);
  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();

    if (last_entry.first == nullptr) {
      continue;
    }
    const Node& n = *last_entry.first;

    if (last_entry.second) {
      // leave node
      if (leave(&n)) {
        return;
      }
      continue;
    }

    if (visited[n.Index()]) continue;

    visited[n.Index()] = true;

    if (enter) {
      if (enter(&n)) {
        return;
      }
    }
    if (leave) stack.emplace_back(&n, true);

    if (comp) {
      InlinedVector<const Node*> sorted_nodes;
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        if (stop && stop(&n, &(*iter))) continue;
        sorted_nodes.push_back(&(*iter));
      }
      std::sort(sorted_nodes.begin(), sorted_nodes.end(), comp);
      for (const auto* in : sorted_nodes) {
        const NodeIndex idx = in->Index();
        if (!visited[idx]) {
          stack.emplace_back(in, false);
        }
      }
    } else {
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        if (stop && stop(&n, &(*iter))) continue;
        const NodeIndex idx = (*iter).Index();
        if (!visited[idx]) {
          stack.emplace_back(graph.GetNode(idx), false);
        }
      }
    }
  }
}

void graph_remove_node(Graph& graph, const NodeInput& node_input) {
  if (node_input.node == nullptr && node_input.node_arg != nullptr) {
    assert(node_input.node_arg->Exists());
    assert(node_arg_is_constant(graph, *node_input.node_arg));
    graph.RemoveInitializedTensor(node_input.node_arg->Name());
  } else if (node_input.node != nullptr && node_input.node_arg != nullptr) {
    graph_remove_node(graph, *node_input.node);
  } else if (node_input.node != nullptr && node_input.node_arg == nullptr) {
    graph_remove_node(graph, *node_input.node);
  } else if (node_input.node == nullptr && node_input.node_arg == nullptr) {
    vai_assert(false, "both node and node_arg are nullptr. not allowed");
  }
}

static std::unique_ptr<Model> clone_model_with_external_data_in_memory(const Graph& original_graph) {
  std::unique_ptr<GraphViewer> graph_viewer = original_graph.CreateGraphViewer();

  const Model& original_model = original_graph.GetModel();
  std::unique_ptr<Model> ret = graph_viewer->CreateModel(
      logging::LoggingManager::DefaultLogger(), original_model.MetaData());

  Graph& graph = ret->MainGraph();

  std::vector<const NodeArg*> graph_input_args;
  for (const NodeArg* arg : graph_viewer->GetInputs()) {
    NodeArg& input_arg = graph.GetOrCreateNodeArg(arg->Name(), arg->TypeAsProto());
    graph_input_args.push_back(&input_arg);
  }
  graph.SetInputs(graph_input_args);

  std::vector<const NodeArg*> graph_output_args;
  for (const NodeArg* arg : graph_viewer->GetOutputs()) {
    NodeArg& output_arg = graph.GetOrCreateNodeArg(arg->Name(), arg->TypeAsProto());
    graph_output_args.push_back(&output_arg);
  }
  graph.SetOutputs(graph_output_args);

  for (auto& it : graph_viewer->GetAllInitializedTensors()) {
    graph_utils::MakeInitializerCopyIfNotExist(original_graph, graph, it.first, false);
  }

  for (const Node& node : graph_viewer->Nodes()) {
    Node& new_node = graph.AddNode(node);
    graph.SetOpSchemaFromRegistryForNode(new_node);
  }

  auto status = graph.Resolve();
  vai_assert(status.IsOK(), " graph resolve error: " + status.ErrorMessage());
  return ret;
}

void graph_save(const Graph& graph, const std::string& filename, const std::string& filename_dat, size_t initializer_size_threshold) {
  std::unique_ptr<Model> model = clone_model_with_external_data_in_memory(graph);
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto;
  if (initializer_size_threshold == std::numeric_limits<size_t>::max()) {
    model_proto = model->ToProto();
  } else {
    ModelSavingOptions model_saving_options{initializer_size_threshold};
    model_proto = model->ToGraphProtoWithExternalInitializers(ToPathString(filename_dat), ToPathString(filename),
                                                              model_saving_options);
  }

  std::fstream output(ToPathString(filename), std::ios::out | std::ios::trunc | std::ios::binary);
  bool result = model_proto->SerializeToOstream(output);
  output << std::flush;
  vai_assert(result, "model serialize to ostream error");
}

vaip_core::DllSafe<std::string> graph_save_string(const Graph& graph) {
  std::unique_ptr<Model> model = clone_model_with_external_data_in_memory(graph);
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto = model->ToProto();

  std::string graph_string;
  bool result = model_proto->SerializeToString(graph_string);
  vai_assert(result, "model serialize to string error");
  return vaip_core::DllSafe(graph_string);
}

Node& graph_fuse(Graph& graph, const std::string& name,
                 const std::string& op_type,
                 const std::vector<size_t>& nodes,
                 const std::vector<std::string>& inputs,
                 const std::vector<std::string>& outputs,
                 const std::vector<std::string>& constant_initializers) {
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->inputs() = inputs;
  meta_def->outputs() = outputs;
  meta_def->constant_initializers() = constant_initializers;
  meta_def->name() = "super_layer";
  meta_def->domain() = "com.xilinx";
  meta_def->since_version() = 1;
  meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;

  auto indexed_subgraph = IndexedSubGraph::Create();
  indexed_subgraph->Nodes() = nodes;
  indexed_subgraph->SetMetaDef(std::move(meta_def));

  auto& fused_node = graph.FuseSubGraph(*indexed_subgraph, name);
  for (auto&& o : fused_node.OutputDefs()) {
    graph.UpdateProducerNode(o->Name(), fused_node.Index());
  }
  return fused_node;
}

Model* model_clone(const Model& original_model, int64_t /*external_data_threshold*/) {
  const Graph& original_graph = const_cast<Model&>(original_model).MainGraph();
  std::unique_ptr<Model> ret = clone_model_with_external_data_in_memory(original_graph);
  return ret.release();
}
}  // namespace vaip
