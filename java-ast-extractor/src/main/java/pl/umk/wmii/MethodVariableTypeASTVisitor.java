package pl.umk.wmii;

import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.List;

public class MethodVariableTypeASTVisitor extends ASTVisitor {

    private final List<String> imports;

    private final List<String> methodVariableTypes;

    public MethodVariableTypeASTVisitor(List<String> imports) {
        this.imports = imports;
        this.methodVariableTypes = new ArrayList<>();
    }

    public List<String> getMethodVariableTypes() {
        return methodVariableTypes;
    }

    @Override
    public boolean visit(VariableDeclarationExpression node) {
        Type type = node.getType();
        addVariableType(type);
        return true;
    }
    @Override
    public boolean visit(VariableDeclarationStatement node) {
        Type type = node.getType();
        addVariableType(type);
        return true;
    }
    @Override
    public boolean visit(SingleVariableDeclaration node) {
        Type type = node.getType();
        addVariableType(type);
        return true;
    }

    private void addVariableType(Type type) {
        ITypeBinding iTypeBinding = type.resolveBinding();
        if (iTypeBinding != null) {
            methodVariableTypes.add(iTypeBinding.getQualifiedName());
        } else {
            String variableType = imports.stream()
                    .filter(importEntry -> importEntry.contains(type.toString()))
                    .findAny()
                    .orElse(type.toString());
            methodVariableTypes.add(variableType);
        }
    }
}
