package pl.umk.wmii;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class SourceFileContent {

    private final Set<String> classNames;
    private final Set<String> superclassNames;
    private final Set<String> interfaceNames;
    private final Set<String> methodNames;
    private final Set<String> variableNames;
    private final List<String> methodContent;
    private List<List<String>> methodVariableTypes;
    private final List<String> commentContent;
    private final String rawSourceContent;

    public SourceFileContent(Set<String> classNames,
            Set<String> superclassNames,
            Set<String> interfaceNames,
            Set<String> methodNames,
            Set<String> variableNames,
            List<String> methodContent,
            List<List<String>> methodVariableTypes,
            List<String> commentContent,
            String rawSourceContent) {
        this.classNames = classNames;
        this.superclassNames = superclassNames;
        this.interfaceNames = interfaceNames;
        this.methodNames = methodNames;
        this.variableNames = variableNames;
        this.methodContent = methodContent;
        this.methodVariableTypes = methodVariableTypes;
        this.commentContent = commentContent;
        this.rawSourceContent = rawSourceContent;
    }
    public Set<String> getClassNames() {
        return classNames;
    }
    public Set<String> getSuperclassNames() {
        return superclassNames;
    }
    public Set<String> getInterfaceNames() {
        return interfaceNames;
    }
    public Set<String> getMethodNames() {
        return methodNames;
    }
    public Set<String> getVariableNames() {
        return variableNames;
    }
    public List<String> getMethodContent() {
        return methodContent;
    }
    public List<List<String>> getMethodVariableTypes() {
        return methodVariableTypes;
    }
    public List<String> getCommentContent() {
        return commentContent;
    }
    public String getRawSourceContent() {
        return rawSourceContent;
    }
}
