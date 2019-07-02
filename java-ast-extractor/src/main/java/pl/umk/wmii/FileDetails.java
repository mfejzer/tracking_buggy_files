package pl.umk.wmii;

public class FileDetails {
    private SourceFileContent source;
    private GraphFeaturesResult graph;

    public FileDetails(SourceFileContent source, GraphFeaturesResult graph) {
        this.source = source;
        this.graph = graph;
    }

    public SourceFileContent getSource() {
        return source;
    }

    public GraphFeaturesResult getGraph() {
        return graph;
    }
}
