package pl.umk.wmii;

import java.util.Map;

public class GraphFeaturesResult {

    private final String className;
    private final String fileName;
    private final Map<String, Integer> dependencies;

    public GraphFeaturesResult(String className, String fileName, Map<String, Integer> dependencies) {
        this.className = className;
        this.fileName = fileName;
        this.dependencies = dependencies;
    }

    public String getClassName() {
        return className;
    }
    public String getFileName() {
        return fileName;
    }
    public Map<String, Integer> getDependencies() {
        return dependencies;
    }
}
