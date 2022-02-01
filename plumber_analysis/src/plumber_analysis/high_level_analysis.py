"""These are functions which are using high-level abstractions.

Not heavily used at the moment, though it would be good to
simplify the API into graph-rewrites and analysis.
"""
import graphsurgeon
import networkx as nx

from plumber_analysis import graphdef_util

class HighLevelPlumberModel(object):
    def __init__(self, plumber):
        self._update_plumber(plumber)

    def _update_plumber(self, plumber):
        self._plumber = plumber
        self._model = self._plumber.model()
        self._graphdef = self._model.graphdef()
        self._recommendation = self._model.recommendation()
        G = graphdef_util.graphdef_to_networkx(self._graphdef, keep_const=False)
        nodes = self._recommendation.ranked_list_bottleneck_nodes_analysis()
        all_attrs = dict()
        for n in nodes:
            attrs = n.to_summary_dict()
            all_attrs[n.name] = attrs
        nx.set_node_attributes(G, attrs)
        self._G = G

    def networkx(self):
        return self._G

    def topologically_mapped_networkx(self):
        G = self.networkx()
        topo_sort = nx.topological_sort(G)
        topo_sort_dataset = filter(graphdef_util.is_dataset_node, topo_sort)
        remapper = graphdef_util.remap_dataset_names(topo_sort_dataset)
        G_remapped = nx.relabel_nodes(G, remapper)
        return G_remapped
