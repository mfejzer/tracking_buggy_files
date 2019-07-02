package pl.umk.wmii;

import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;

public class GraphFeaturesAstWalker {

    private final ASTParser parser = ASTParser.newParser(AST.JLS8);

    public GraphFeaturesResult extract(String filename, String rawSource) {

        parser.setSource(rawSource.toCharArray());
        parser.setKind(ASTParser.K_COMPILATION_UNIT);

        //Add empty project environment to resolve bindings for method variables
        parser.setResolveBindings(true);
        String[] emptyStringArray = new String[0];
        parser.setEnvironment(emptyStringArray, emptyStringArray, emptyStringArray, true);
        parser.setUnitName("dontCare");

        final CompilationUnit cu = (CompilationUnit) parser.createAST(null);

        GraphFeaturesASTVisitor graphFeaturesASTVisitor = new GraphFeaturesASTVisitor(filename);
        cu.accept(graphFeaturesASTVisitor);

        return graphFeaturesASTVisitor.getResult();
    }
}
