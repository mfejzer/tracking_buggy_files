package pl.umk.wmii;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MissingGraphNotesApp {

    public static void main(String[] args) throws IOException, InterruptedException {
        String repositoryPath = args[0];
        String bugFixSha = args[1];

        final GraphFeaturesAstWalker astWalker = new GraphFeaturesAstWalker();

        final ObjectMapper objectMapper = new ObjectMapper();

        IncrementalGraphNotesApp.createNotes(astWalker, objectMapper, repositoryPath, bugFixSha);


    }

}
