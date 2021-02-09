import numpy as np
import itertools
import sympy as sym

class LabelledGraph:
    """
    A class representing the graph G= (V, E = U_s E^s) obtained by fusing the graphs
    G^s = (V, E^s) of each sub-swarm.
    The initialization of the class requires only the set of nodes V
    which corresponds to the bins  repartrion
    :param V : a set of node names, the type doesn't matter as long as the names
               are used consistently
    """

    def __init__(self, V):
        # Initialize the set of nodes
        self.V = set(V)
        # Initialize the set of edge to be an empty set
        self.E = set()
        # Store the number of bins
        self.nr = len(self.V)
        # Initialize the number of sub-swarm to be 0
        self.m = 0
        # Initialize the dictionary containing the node labels as symbolic expressions
        self.nLabels = dict()
        # Initialize the dictionary containing a tuple (coefficients, indexes)
        self.nLabelsRepr = dict()
        # Initialize the dictionary containing the edges of the each sub-swarm
        self.eSubswarm = dict()
        # Initialize the directory containing the set of non-existing edges for subswarms
        self.neSubswarm = dict()
        # Initialize the template for symbolic variable name
        self.templateDensity = 'x_{}_{}'

    def addSubswarm(self, nameSubswarm, Es):
        """
        Add a subswarm given by the name 'nameSubswarm' and the transitions
        of the agents are given by the edge 'Es'.

        In order to access the symbolic variables defining the density
        distribution of this swarm, one must use the attribute symbolic value
        self.x_{nameSubswarm}_{nodeLabel} denoting the density of the
        subswarm 'nameSubswarm' at node 'nodeLabel'.

        :param nameSubswarm : Name of the subswarm
        :param Es : The edges encoding the transitions of the subswarm
        """
        assert nameSubswarm not in self.eSubswarm, \
                "Subswarm {} already present in the graph".format(nameSubswarm)
        Es = set(Es)
        # Check if the nodes in the edge set are in the set of node V
        for (n1,n2) in Es:
            assert n1 in self.V and n2 in self.V,\
                    "The node {} or {} is not present in the graph".format(n1,n2)
        self.E = self.E | Es # Add the edge set the the global edge set
        self.eSubswarm[nameSubswarm] = Es # Save the edge set for this subswarm
        self.neSubswarm[nameSubswarm] = set()
        # self.AaSubswarm[nameSubswarm] = np.zeros((self.nr,self.nr)) # Initialize the adjacecny matrix
        # Find out the nodes that are not connected together by an edge
        for n1 in self.V:
            for n2 in self.V:
                if (n1,n2) not in Es:
                    self.neSubswarm[nameSubswarm].add((n1,n2))
        # Update the number of subswarm with motions in the graph
        self.m = self.m + 1
        # Create the symbolic variables to use in order to label the nodes of the graph
        for nName in self.V:
            varName = self.templateDensity.format(nameSubswarm, nName)
            setattr(self, varName, sym.symbols(varName))


    def addNodeLabel(self, nodeName, nodeLabel):
        """
        Set the label of the node 'nodeName' to be the mathematical function
        'nodeLabel'. Noe that nodeLabel is an expression involving the density
        distribution of the subswarms
        """
        assert nodeName in self.V and nodeName not in self.nLabels,\
            "The node {} is either not present in the graph or already has been labelled.".format(nodeName)
        self.nLabels[nodeName] = nodeLabel
        self.nLabelsRepr[nodeName] = self.getMatReprNodeLabel(nodeLabel)

    def getGlobalEdgeSet(self):
        """
        Return the adjacency matrix corresponding to the union of all the subswarms
        adjacency matrices
        """
        return self.E

    def getSubswarmEdgeSet(self, nameSubswarm):
        """
        Return the edge se that encodes the motion constraints of the
        subswarm whose identifier is 'nameSubswarm'
        """
        assert nameSubswarm in self.eSubswarm,\
                "Subswarm {} not present in the graph".format(nameSubswarm)
        return self.eSubswarm[nameSubswarm]

    def getSubswarmNonEdgeSet(self, nameSubswarm):
        """
        Return the set of nodes (n1,n2) such that (n1,n2) are not present in the edge
        set of the subswarm whose identifier is 'nameSubswarm'
        """
        assert nameSubswarm in self.neSubswarm,\
                "Subswarm {} not present in the graph".format(nameSubswarm)
        return self.neSubswarm[nameSubswarm]

    def getNodeLabel(self, nodeName):
        """
        Return the node label associated to the node identifier 'nodeName'
        :param nodeName : The name of the node to provide the label
        """
        assert nodeName in self.V and nodeName in self.nLabels,\
                "Node {} is not present in either the graph or has not been labelled yet".format(nodeName)
        return self.nLabels[nodeName]

    def getLabelMatRepr(self, nodeName):
        """
        Return a matrix representation of the label of nodeName
        """
        return self.nLabelsRepr.get(nodeName, None)

    def getMatReprNodeLabel(self, nodeLabel):
        """
        Return a matrix representation of the constraints given by nodeLabel
        :param nodeLabel : a symbolic expression for node labels
        """
        coeffList = list()
        indexList = list()
        try:
            iterator = iter(nodeLabel)
        except TypeError:
            iterator = [nodeLabel]
        for nLabel in iterator:
            cList = list()
            iList = list()
            # Build the polynomial from the symbolic expression nodeLabel
            polyNodeLabel = sym.Poly(nLabel)
            # Get the symbols from the polynomial expression
            varExpr = polyNodeLabel.gens
            for varname in varExpr:
                # Get the coefficient corresponding to varname
                cV = float(polyNodeLabel.coeff_monomial(varname))
                # Split varname to obtain the node and subswarm name
                splitVal = str(varname).split('_')
                sId = splitVal[1]
                nId = splitVal[2]
                # Iterate over the node to get the actual node identifier
                for v in self.V:
                    if str(v) == nId:
                        nId = v
                        break
                # Iterate over the subswarm to get the actual subswarm identifier
                for s in self.eSubswarm.keys():
                    if str(s) == sId:
                        sId = s
                        break
                indexV = (sId, nId)
                # Add the coefficients to the list
                cList.append(cV)
                iList.append(indexV)
            coeffList.extend(cList)
            indexList.extend(iList)
        return (coeffList, indexList)

    def __repr__(self):
        s0 = "----------------------------------------------------------------\n"
        s1 = "A labelled graph G defined as follows: \n"
        s2 = "  Set of nodes: V = {}\n".format(self.V)
        s3 = "  Set of subswarm labels : {}\n".format(self.eSubswarm.keys())
        s4 = "  Set of edge in the union graph : {}\n".format(self.E)
        s5 = ""
        for key, value in self.eSubswarm.items():
            s5 = s5 + "  Set of edge for subswarm {} : {}\n".format(key, value)
        s6 = ""
        for key, value in self.neSubswarm.items():
            s6 = s6 + "  Set of non-existing edge for subswarm {} : {}\n".format(key, value)
        s7 = "  Number of nodes: {}\n".format(self.nr)
        s8 = "  Number of subswarms {}\n".format(self.m)
        s9 = ""
        for key, value in self.nLabels.items():
            s9 = s9 + "  Label of the node {} is {}\n".format(key, value)
        return s0 + s1 + s7 + s8 + s2 + s3 + s4 + s5 + s6 + s9
