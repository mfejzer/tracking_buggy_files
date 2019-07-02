package pl.umk.wmii;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class GraphFeaturesAstWalkerTest {

    @Test
    public void shouldResolvePackageName() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getClassName()).contains("com.Example");
    }

    @Test
    public void shouldHandleNotStaticImports() {
        String source = "package com;\n" +
                "import org.assertj.core.api.AssertionInfo;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.assertj.core.api.AssertionInfo");
    }

    @Test
    public void shouldHandleStaticImports() {
        String source = "package com;\n" +
                "import static org.assertj.core.api.Assertions.assertThat;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.assertj.core.api.Assertions");
    }

    @Test
    public void shouldHandleMethodReturnTypesFromOtherPackages() {
        String source = "package com;\n" +
                "import org.Example2;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static Example2 method1() {\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }

    @Test
    public void shouldHandleExplicitMethodReturnTypesFromOtherPackages() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static org.Example2 method1() {\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }

    @Test
    public void shouldHandleMethodReturnTypesFromTheSamePackage() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static Example2 method1() {\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("com.Example2");
    }

    @Test
    public void shouldHandleClassFieldFromDifferentPackage() {
        String source = "package com;\n" +
                "import org.Example2;\n" +
                "\n" +
                "public class Example \n" +
                "    private Example2 example2Instance = null;\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }

    @Test
    public void shouldHandleExplicitClassFieldFromDifferentPackage() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "    private org.Example2 example2Instance = null;\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }

    @Test
    public void shouldHandleClassFieldFromTheSamePackage() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "    private Example2 example2Instance = null;\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("com.Example2");
    }

    @Test
    public void shouldHandleVariableFromDifferentPackage() {
        String source = "package com;\n" +
                "import org.Example2;\n" +
                "\n" +
                "public class Example \n" +
                "    public static void method1() {\n" +
                "        Example2 example2Instance = null;\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }

    @Test
    public void shouldHandleExplicitVariableFromDifferentPackage() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "    public static void method1() {\n" +
                "        org.Example2 example2Instance = null;\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }

    @Test
    public void shouldHandleVariableFromTheSamePackage() {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "    public static void method1() {\n" +
                "        Example2 example2Instance = null;\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("com.Example2");
    }

    @Test
    public void shouldHandleStaticMethodInvocation() {
        String source = "package com;\n" +
                "import static org.Example2.method2;\n" +
                "\n" +
                "public class Example \n" +
                "    public static void method1() {\n" +
                "        Example2.method2();\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }
    @Test
    public void shouldHandleNotStaticMethodInvocation() {
        String source = "package com;\n" +
                "import org.Example2;\n" +
                "\n" +
                "public class Example \n" +
                "    public static void method1() {\n" +
                "        Example2 instance = new Example2();\n" +
                "        instance.method2();\n" +
                "    }\n" +
                "\n" +
                "}";
        GraphFeaturesAstWalker walker = new GraphFeaturesAstWalker();

        GraphFeaturesResult result = walker.extract("src/main/java/com/Example.java", source);

        assertThat(result.getDependencies()).containsKey("org.Example2");
    }
}