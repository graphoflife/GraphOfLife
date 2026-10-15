"""The world statistics a lab run records, grouped by what they measure: a correlation inside a group is
mostly a matter of definition, one between groups says something about how the world works."""
GROUPS = {
    "inequality": {"gini", "tokenEvenness", "tokenEntropy", "topDecileShare", "maxTokens", "minTokens", "medianTokens",
                   "tokenExponent", "tokenExponentR2"},
    "size": {"nodes", "edges", "meanTokens", "density", "meanDegree", "degreeEntropy", "degreeEvenness", "medianDegree"},
    "flow": {"totalFlow", "selfAllocationShare", "meanEdgeFlow", "maxEdgeFlow", "spreadShare", "heldHomeShare"},
    "cycling": {"cyclingShare", "netCyclingShare", "lightningScore", "lightningLongest", "netLightningScore",
                "netLightningLongest", "netFlowShare", "flowImbalance"},
    "degree tail": {"degreeGamma", "degreeKMin", "degreeTailShare", "degreeGammaKS", "degreeGammaR2", "degreeExponent",
                    "degreeExponentR2", "maxDegree"},
    "fragility": {"bridges", "cutRisk", "cutRiskBefore", "coreShare", "leaves", "minDegree", "components", "orphaned"},
    "loops": {"transitivity", "triangles", "trianglesVsDegree", "trianglesVsDegreeR2", "clusteringVsDegree",
              "clusteringVsDegreeR2", "cycleRank", "loopDensity"},
    "shape": {"dimension", "boxDimension", "boxDimensionR2", "diameter", "radius", "meanPathLength", "spectralGap",
              "ricciCurvature", "assortativity"},
    "brains": {"brainDiversity", "distinctBrains", "distinctParents", "cladesInWindow"},
    "change": {"maxTokenAdded", "maxTokenLost", "gainers", "losers", "changeVsTokens", "changeVsTokensR2",
               "tokensVsDegree", "tokensVsDegreeR2"},
    "life": {"births", "starved", "redistributed", "meanChildLinks", "meanInvestedShare", "reproTokenShare",
             "handovers", "prunedEdges"},
    "fighting": {"revolutions", "revoltShare"},
    "gifts": {"gifts", "giftTokens", "giftShare"},
}
GROUP_OF = {name: g for g, names in GROUPS.items() for name in names}
