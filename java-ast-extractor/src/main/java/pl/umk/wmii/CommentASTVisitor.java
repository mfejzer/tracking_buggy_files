package pl.umk.wmii;

import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.List;

public class CommentASTVisitor extends ASTVisitor {

    private final String rawSource;

    private final List<String> commentContent;

    public CommentASTVisitor(String rawSource) {
        this.rawSource = rawSource;
        this.commentContent = new ArrayList<>();
    }

    public List<String> getCommentContent() {
        return commentContent;
    }
    public boolean visit(BlockComment blockComment) {
        addComment(blockComment);
        return true;
    }

    public boolean visit(LineComment lineComment) {
        addComment(lineComment);
        return true;
    }

    public boolean visit(Javadoc javadoc) {
        addComment(javadoc);
        return true;
    }

    private void addComment(Comment commentAstNode) {
        final int startPosition = commentAstNode.getStartPosition();
        final int endPosition = startPosition + commentAstNode.getLength();
        final String content = rawSource.substring(startPosition, endPosition);
        commentContent.add(content);
    }

}
