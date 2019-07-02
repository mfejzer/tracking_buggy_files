package pl.umk.wmii;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class IncrementalNotesApp {

    public static void main(String[] args) throws IOException, InterruptedException {
        String repositoryPath = args[0];
        String jsonFilePath = args[1];

        final AstWalker astWalker = new AstWalker();

        final ObjectMapper objectMapper = new ObjectMapper();

        List<String> commitsSortedByDate = CommitLoader.getCommitsSortedByDate(jsonFilePath, objectMapper);

        commitsSortedByDate.forEach(commit -> {
            try {
                createNotes(astWalker, objectMapper, repositoryPath, commit);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

    }

    private static void createNotes(AstWalker astWalker, ObjectMapper objectMapper, String repositoryPath, String commit) throws IOException, InterruptedException {
        String parent = commit + "^";
        List<String> files = Git.lsTree(repositoryPath, parent);
        List<String[]> shaAndFilenames = files.stream()
                .filter(s -> s.endsWith(".java"))
                .map(Git::extractShaAndFilename)
                .collect(Collectors.toList());

        Set<String> fileShas = shaAndFilenames.stream().map(r -> r[0]).collect(Collectors.toSet());

        List<String> noteShaFileShaList = Git.notesList(repositoryPath);
        Set<String> alreadyNoted = noteShaFileShaList.stream().map(line -> line.split(" ")[1]).collect(Collectors.toSet());

        fileShas.removeAll(alreadyNoted);

        fileShas.forEach(fileSha -> {
            try {
                addNote(astWalker, objectMapper, repositoryPath, fileSha);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    private static void addNote(AstWalker astWalker, ObjectMapper objectMapper, String repositoryPath, String fileSha) throws IOException, InterruptedException {
        String fileContent = Git.catFile(repositoryPath, fileSha);
        SourceFileContent sourceFileContent = astWalker.extract(fileContent);
        String note = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(sourceFileContent);
        Git.addNote(repositoryPath, fileSha, note);
    }

}
