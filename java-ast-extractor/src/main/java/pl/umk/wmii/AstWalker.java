package pl.umk.wmii;

import org.eclipse.jdt.core.dom.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class AstWalker {

    private final ASTParser parser = ASTParser.newParser(AST.JLS8);

    public SourceFileContent extract(String rawSource) {

        parser.setSource(rawSource.toCharArray());
        parser.setKind(ASTParser.K_COMPILATION_UNIT);

        //Add empty project environment to resolve bindings for method variables
        parser.setResolveBindings(true);
        String[] emptyStringArray = new String[0];
        parser.setEnvironment(emptyStringArray, emptyStringArray, emptyStringArray, true);
        parser.setUnitName("dontCare");

        final CompilationUnit cu = (CompilationUnit) parser.createAST(null);

        final Set<String> classNames = new HashSet<>();
        final Set<String> interfaceNames = new HashSet<>();
        final Set<String> superclassNames = new HashSet<>();
        final Set<String> methodNames = new HashSet<>();
        final Set<String> variableNames = new HashSet<>();
        final List<String> methodContent = new ArrayList<>();
        final List<List<String>> methodVariableTypes = new ArrayList<>();
        final List<String> imports = new ArrayList<>();

        cu.accept(new ASTVisitor(true) {

            @Override
            public boolean visit(TypeDeclaration typeDeclaration) {
                if (typeDeclaration.resolveBinding() != null) {
                    classNames.add(typeDeclaration.resolveBinding().getQualifiedName());
                } else {
                    classNames.add(typeDeclaration.getName().getFullyQualifiedName());
                }

                final Type rawSuperclassType = typeDeclaration.getSuperclassType();
                if (rawSuperclassType != null && rawSuperclassType instanceof SimpleType) {
                    final SimpleType superclassType = (SimpleType) rawSuperclassType;
                    String fullyQualifiedName = superclassType.getName().getFullyQualifiedName();
                    String superclassName = imports.stream()
                            .filter(importEntry -> importEntry.contains(fullyQualifiedName))
                            .findAny()
                            .orElse(fullyQualifiedName);
                    superclassNames.add(superclassName);
                }

                typeDeclaration.superInterfaceTypes().forEach(superInterface -> {
                    if (superInterface instanceof SimpleType) {
                        final SimpleType interfaceType = (SimpleType) superInterface;
                        String fullyQualifiedName = interfaceType.getName().getFullyQualifiedName();
                        String interfaceName = imports.stream()
                                .filter(importEntry -> importEntry.contains(fullyQualifiedName))
                                .findAny()
                                .orElse(fullyQualifiedName);
                        interfaceNames.add(interfaceName);
                    }
                });

                return true;
            }

            @Override
            public boolean visit(MethodDeclaration methodDeclaration) {
                final SimpleName name = methodDeclaration.getName();
                methodNames.add(name.getIdentifier());

                MethodVariableTypeASTVisitor methodVariableTypeASTVisitor = new MethodVariableTypeASTVisitor(imports);
                methodDeclaration.accept(methodVariableTypeASTVisitor);
                methodVariableTypes.add(methodVariableTypeASTVisitor.getMethodVariableTypes());

                int startPosition = methodDeclaration.getStartPosition();
                int endPosition = startPosition + methodDeclaration.getLength();
                String content = rawSource.substring(startPosition, endPosition);
                methodContent.add(content);
                return true;
            }

            @Override
            public boolean visit(VariableDeclarationFragment variableDeclarationFragment) {
                final SimpleName name = variableDeclarationFragment.getName();
                variableNames.add(name.getIdentifier());
                return true;
            }

            @Override
            public boolean visit(ImportDeclaration node) {
                imports.add(node.getName().getFullyQualifiedName());
                return true;
            }
        });

        final CommentASTVisitor commentAstVisitor = new CommentASTVisitor(rawSource);

        cu.getCommentList().forEach(commentObject -> {
            Comment comment = (Comment) commentObject;
            comment.accept(commentAstVisitor);
        });

        final List<String> commentContent = commentAstVisitor.getCommentContent();

        return new SourceFileContent(classNames,
                superclassNames,
                interfaceNames,
                methodNames,
                variableNames,
                methodContent,
                simplifyListGenerics(methodVariableTypes),
                commentContent,
                rawSource);
    }

    private List<List<String>> simplifyListGenerics(List<List<String>> methodVariableTypes) {
        return methodVariableTypes.stream().map(this::simplifyGenerics).collect(Collectors.toList());
    }

    private List<String> simplifyGenerics(List<String> variableTypes) {
        Set<String> typesSet = variableTypes
                .stream()
                .flatMap(variableType -> {
                    if (variableType.contains("<")) {
                        String[] splitResult = variableType
                                .replace('<', ' ')
                                .replace('>', ' ')
                                .replace(',', ' ')
                                .split(" ");
                        return Arrays.stream(splitResult).filter(s -> !s.isEmpty());
                    } else {
                        return Stream.of(variableType);
                    }
                })
                .map(variableType -> variableType.replaceAll("\\[", "").replaceAll("\\]", ""))
                .collect(Collectors.toSet());

        return new ArrayList<>(typesSet);
    }
}
