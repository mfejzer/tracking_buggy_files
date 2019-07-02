package pl.umk.wmii;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class GitTest {

    @Test
    public void shouldExtractShaAndFilenameFromLineWithSpace()  {
        String line = "100644 blob e365adb169e5d260506d2dfc3a9d5f91a57e080d\tbundles/org.eclipse.ui.workbench/Eclipse UI/org/eclipse/ui/internal/PageSelectionService.java";

        String[] result = Git.extractShaAndFilename(line);

        assertThat(result[0]).contains("e365adb169e5d260506d2dfc3a9d5f91a57e080d");
        assertThat(result[1]).contains("bundles/org.eclipse.ui.workbench/Eclipse UI/org/eclipse/ui/internal/PageSelectionService.java");
    }

}