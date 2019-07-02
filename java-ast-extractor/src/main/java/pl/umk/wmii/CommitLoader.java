package pl.umk.wmii;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

public class CommitLoader {

    static List<String> getCommitsSortedByDate(String jsonFilePath, ObjectMapper objectMapper) throws IOException {
        byte[] jsonData = Files.readAllBytes(Paths.get(jsonFilePath));
        JsonNode rootNode = objectMapper.readTree(jsonData);

        SimpleDateFormat dateFormat = new SimpleDateFormat("EEE MMM dd HH:mm:ss yyyy Z", Locale.US);

        Map<String, Date> commitDates = new HashMap<>();

        rootNode.fields().forEachRemaining(kv -> {
            String commit = kv.getValue().path("commit").path("metadata").path("sha").textValue().substring(7).trim();
            String rawDate = kv.getValue().path("commit").path("metadata").path("date").textValue().substring(6).trim();

            try {
                Date date = dateFormat.parse(rawDate);
                commitDates.put(commit, date);
            } catch (ParseException e) {
                throw new RuntimeException(e);
            }
        });

        return commitDates.entrySet().stream().sorted(Map.Entry.comparingByValue()).map(Map.Entry::getKey).collect(Collectors.toList());
    }
}
