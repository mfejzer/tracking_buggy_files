package pl.umk.wmii;

import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class GraphFeaturesASTVisitor extends ASTVisitor {

    private String className;
    private String packageName;
    private final String fileName;
    private final List<String> imports;
    private final List<String> dependencies;

    public GraphFeaturesASTVisitor(String fileName) {
        super(false);
        this.fileName = fileName;
        this.imports = new ArrayList<>();
        this.dependencies = new ArrayList<>();
    }

    @Override
    public boolean visit(TypeDeclaration typeDeclaration) {
        if (typeDeclaration.resolveBinding() != null) {
            className = typeDeclaration.resolveBinding().getQualifiedName();
            packageName = typeDeclaration.resolveBinding().getPackage().getName();
        } else {
            className = typeDeclaration.getName().getFullyQualifiedName();
            if (className.contains(".")) {
                int classNameIndex = className.lastIndexOf('.');
                packageName = className.substring(0, classNameIndex);
            } else {
                packageName = "";
            }
        }
        return true;
    }

    @Override
    public boolean visit(ImportDeclaration importDeclaration) {
        String fullyQualifiedName = importDeclaration.getName().getFullyQualifiedName();
        if (fullyQualifiedName.contains("*")) {
            return true;
        }
        if (importDeclaration.isStatic()) {
            int methodNameIndex = fullyQualifiedName.lastIndexOf('.');
            String importClassName = fullyQualifiedName.substring(0, methodNameIndex);
            imports.add(importClassName);
        } else {
            imports.add(fullyQualifiedName);
        }
        return true;
    }

    @Override
    public boolean visit(SingleVariableDeclaration declaration) {
        if (declaration.getType().isSimpleType()) {
            String fullyQualifiedName = ((SimpleType) declaration.getType()).getName().getFullyQualifiedName();
            String dependency = resolveDependencyWithImports(fullyQualifiedName);
            dependencies.add(dependency);
        }

        return true;
    }

    @Override
    public boolean visit(MethodInvocation methodInvocation) {
        Expression expression = methodInvocation.getExpression();
        if (expression != null && SimpleType.class.isInstance(expression)) {
            String fullyQualifiedName = ((SimpleName) methodInvocation.getExpression()).getIdentifier();
            String dependency = resolveDependencyWithImports(fullyQualifiedName);
            dependencies.add(dependency);
        }
        return true;
    }

    @Override
    public boolean visit(FieldDeclaration declaration) {
        if (declaration.getType().isSimpleType()) {
            String fullyQualifiedName = ((SimpleType) declaration.getType()).getName().getFullyQualifiedName();
            String dependency = resolveDependencyWithImports(fullyQualifiedName);
            dependencies.add(dependency);
        }

        return true;
    }

    @Override
    public boolean visit(ClassInstanceCreation classInstanceCreation) {
        if (classInstanceCreation.getType().isSimpleType()) {
            String fullyQualifiedName = ((SimpleType) classInstanceCreation.getType()).getName().getFullyQualifiedName();
            String dependency = resolveDependencyWithImports(fullyQualifiedName);
            dependencies.add(dependency);
        }
        return true;
    }

    @Override
    public boolean visit(SimpleType simpleType) {
        String fullyQualifiedName = simpleType.getName().getFullyQualifiedName();
        String dependency = resolveDependencyWithImports(fullyQualifiedName);
        dependencies.add(dependency);
        return true;
    }

    private String resolveDependencyWithImports(String fullyQualifiedName) {
        if (fullyQualifiedName.contains(".")) {
            return fullyQualifiedName;
        }
        return imports.stream()
                .filter(i -> i.contains(fullyQualifiedName))
                .findAny().orElseGet(() -> prepare(fullyQualifiedName));
    }
    private String prepare(String fullyQualifiedName) {
        if (packageName == null || packageName.equals("")) {
            return fullyQualifiedName;
        }
        return packageName + "." + fullyQualifiedName;
    }

    public GraphFeaturesResult getResult() {
        Map<String, Integer> result = new HashMap<>();
        imports.forEach(insert(result));
        dependencies.forEach(insert(result));
        return new GraphFeaturesResult(className, fileName, result);
    }
    private Consumer<String> insert(Map<String, Integer> result) {
        return entry -> {
            result.putIfAbsent(entry, 0);
            Integer count = result.get(entry);
            count += 1;
            result.put(entry, count);
        };
    }
}
