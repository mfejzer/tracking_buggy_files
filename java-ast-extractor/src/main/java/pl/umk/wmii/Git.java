package pl.umk.wmii;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.List;
import java.util.stream.Collectors;

public class Git {

    public static List<String> lsTree(String repositoryPath, String sha) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("git", "-C", repositoryPath,
                "ls-tree", "-r", sha);

        return runRetrievalProcess(processBuilder);
    }

    static String[] extractShaAndFilename(String lsTreeResultLine){
        String[] parts = lsTreeResultLine.split("\t");
        String filename = parts[1];
        String sha = parts[0].split(" ")[2];
        return new String[]{sha, filename};
    }

    public static String catFile(String repositoryPath, String fileSha) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("git", "-C", repositoryPath,
                "cat-file", "blob", fileSha);

        final Process process = processBuilder.start();
        return new BufferedReader(new InputStreamReader(process.getInputStream()))
                .lines()
                .collect(Collectors.joining("\n"));
    }

    public static List<String> notesList(String repositoryPath) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("git", "-C", repositoryPath,
                "notes", "list");

        return runRetrievalProcess(processBuilder);
    }

    public static List<String> notesListWithRef(String repositoryPath, String ref) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("git", "-C", repositoryPath,
                "notes",
                "--ref", ref,
                "list");

        return runRetrievalProcess(processBuilder);
    }

    private static List<String> runRetrievalProcess(ProcessBuilder processBuilder) throws IOException, InterruptedException {
        final Process process = processBuilder.start();
        final List<String> result = new BufferedReader(new InputStreamReader(process.getInputStream()))
                .lines()
                .collect(Collectors.toList());
        process.waitFor();
        return result;
    }

    public static void addNote(String repositoryPath, String fileSha, String note) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("git", "-C", repositoryPath, "notes",
                "add", "-F", "-", fileSha);

        runInserProcess(note, processBuilder);

    }

    public static void addNoteWithRef(String repositoryPath, String fileSha, String note, String ref) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("git", "-C", repositoryPath,
                "notes",
                "--ref", ref,
                "add", "-F", "-", fileSha);

        runInserProcess(note, processBuilder);

    }

    private static void runInserProcess(String note, ProcessBuilder processBuilder) throws IOException, InterruptedException {
        final Process process = processBuilder.start();
        final PrintWriter printWriter = new PrintWriter(process.getOutputStream());
        printWriter.print(note);
        printWriter.close();

        process.waitFor();
    }
}
