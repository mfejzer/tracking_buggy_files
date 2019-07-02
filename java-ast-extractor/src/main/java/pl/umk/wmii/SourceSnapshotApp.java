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


public class SourceSnapshotApp {

    public static List<Path> findAllSourceFilesInRepository(String repositoryPath) throws IOException {
        return Files.find(Paths.get(repositoryPath), 100,
                (path, attributes) -> path.toString().toLowerCase().endsWith(".java"))
                .collect(Collectors.toList());
    }

    public static void main(String[] args) throws IOException {
        final AstWalker astWalker = new AstWalker();
        final GraphFeaturesAstWalker graphFeaturesAstWalker = new GraphFeaturesAstWalker();

        String sourcePath = args[0];
        String outputFile = args[1];

        ObjectMapper objectMapper = new ObjectMapper();

        final Map<String, FileDetails> contentMap = new HashMap<>();

        findAllSourceFilesInRepository(sourcePath).forEach(path -> {
            try {
                String rawContent = new String(Files.readAllBytes(path));
                SourceFileContent sourceFileContent = astWalker.extract(rawContent);
                GraphFeaturesResult graphFeaturesResult = graphFeaturesAstWalker.extract(path.getFileName().toString(), rawContent);
                FileDetails fileDetails = new FileDetails(sourceFileContent, graphFeaturesResult);
                contentMap.put(path.toString(), fileDetails);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });

        objectMapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputFile), contentMap);

    }

}
