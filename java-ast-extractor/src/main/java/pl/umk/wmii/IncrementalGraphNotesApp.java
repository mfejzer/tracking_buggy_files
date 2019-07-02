package pl.umk.wmii;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.lang.String.format;

public class IncrementalGraphNotesApp {

    public static void main(String[] args) throws IOException, InterruptedException {
        final DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        System.out.println(format("Started: %s", dateFormat.format(new Date())));

        String repositoryPath = args[0];
        String jsonFilePath = args[1];

        final GraphFeaturesAstWalker astWalker = new GraphFeaturesAstWalker();

        final ObjectMapper objectMapper = new ObjectMapper();

        List<String> commitsSortedByDate = CommitLoader.getCommitsSortedByDate(jsonFilePath, objectMapper);

        commitsSortedByDate.forEach(commit -> {
            try {
                System.out.println(format("Processing %s", commit));
                createNotes(astWalker, objectMapper, repositoryPath, commit);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        System.out.println(format("Finished: %s", dateFormat.format(new Date())));
    }

    public static void createNotes(GraphFeaturesAstWalker astWalker, ObjectMapper objectMapper, String repositoryPath, String commit) throws IOException, InterruptedException {
        String parent = commit + "^";
        List<String> files = Git.lsTree(repositoryPath, parent);
        List<String[]> shaAndFilenames = files.stream()
                .filter(s -> s.endsWith(".java"))
                .map(Git::extractShaAndFilename)
                .collect(Collectors.toList());

        final Map<String, String> shaToFileName = shaAndFilenames.stream()
                .collect(Collectors.toMap(r -> r[0], r -> r[1], (a, b)-> a));
        Set<String> fileShas = shaAndFilenames.stream().map(r -> r[0]).collect(Collectors.toSet());

        List<String> noteShaFileShaList = Git.notesListWithRef(repositoryPath, "refs/notes/graph");
        Set<String> alreadyNoted = noteShaFileShaList.stream().map(line -> line.split(" ")[1]).collect(Collectors.toSet());

        fileShas.removeAll(alreadyNoted);

        fileShas.forEach(fileSha -> {
            try {
                addNote(astWalker, objectMapper, repositoryPath, fileSha, shaToFileName);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    private static void addNote(GraphFeaturesAstWalker astWalker, ObjectMapper objectMapper, String repositoryPath, String fileSha, final Map<String, String> shaToFileName) throws IOException, InterruptedException {
        String fileContent = Git.catFile(repositoryPath, fileSha);
        String fileName = shaToFileName.get(fileSha);
        GraphFeaturesResult result = astWalker.extract(fileName, fileContent);
        String note = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
        Git.addNoteWithRef(repositoryPath, fileSha, note, "refs/notes/graph");
    }

}
