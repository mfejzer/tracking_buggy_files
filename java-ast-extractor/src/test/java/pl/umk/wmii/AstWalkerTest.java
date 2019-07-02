package pl.umk.wmii;

import org.junit.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

public class AstWalkerTest {

    private AstWalker astWalker = new AstWalker();

    @Test
    public void shouldExtractClassesMethodsAndVariables() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import org.junit.Test;\n" +
                "\n" +
                "public class Example {\n" +
                "\n" +
                "    public void method1() {\n" +
                "        String variable1 = null;\n" +
                "        String variable2 = null;\n" +
                "    }\n" +
                "    public static void method2() {\n" +
                "        String variable3 = null;\n" +
                "        String variable4 = null;\n" +
                "    }\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getClassNames()).contains("com.Example");
        assertThat(sourceFileContent.getMethodNames()).contains("method1", "method2");
        assertThat(sourceFileContent.getVariableNames()).contains("variable1", "variable2", "variable3", "variable4");

    }

    @Test
    public void shouldExtractSuperclassAndIterfaceas() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import org.junit.Test;\n" +
                "\n" +
                "public class Example extends Superclass implements Interface1, Interface2{\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getClassNames()).contains("com.Example");
        assertThat(sourceFileContent.getSuperclassNames()).contains("Superclass");
        assertThat(sourceFileContent.getInterfaceNames()).contains("Interface1", "Interface2");
    }

    @Test
    public void shouldResolveSuperclassPackageFromImports() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import org.junit.Test;\n" +
                "import org.Superclass;\n" +
                "\n" +
                "public class Example extends Superclass {\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getClassNames()).contains("com.Example");
        assertThat(sourceFileContent.getSuperclassNames()).contains("org.Superclass");
    }

    @Test
    public void shouldResolveInterfacePackageFromImports() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import org.junit.Test;\n" +
                "import org.Interface;\n" +
                "\n" +
                "public class Example implements Interface {\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getClassNames()).contains("com.Example");
        assertThat(sourceFileContent.getInterfaceNames()).contains("org.Interface");
    }

    @Test
    public void shouldExtractMethodContent() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "        String variable1 = null;\n" +
                "        Integer variable2 = null;\n" +
                "    }\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getMethodContent().contains(
                "public static void method1() {\n" +
                        "        String variable1 = null;\n" +
                        "        Integer variable2 = null;\n" +
                        "    }"));
        assertThat(sourceFileContent.getMethodVariableTypes().contains(Arrays.asList("String", "Integer")));

    }

    @Test
    public void shouldExtractCommentContent() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "/**\n" +
                " * Javadoc line 1\n" +
                " * Javadoc line 2\n" +
                " */\n" +
                "public class Example \n" +
                "\n" +
                "    //comment inside class\n" +
                "    public static void method1() {\n" +
                "        String variable1 = null;\n" +
                "        //comment inside method\n" +
                "        String variable2 = null;\n" +
                "    }\n" +
                "/*\n" +
                "another comment\n" +
                " */\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getCommentContent().size()).isEqualTo(4);
        assertThat(sourceFileContent.getCommentContent()).contains("//comment inside class");
        assertThat(sourceFileContent.getCommentContent()).contains("//comment inside method");

    }

    @Test
    public void shouldContainNotModifiedSource() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "        String variable1 = null;\n" +
                "    }\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getRawSourceContent()).contains(source);
    }

    @Test
    public void shouldResolveMethodVariableTypes() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import java.util.List;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "        List<String> variable1 = null;\n" +
                "    }\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getMethodVariableTypes().get(0)).contains("java.util.List", "java.lang.String");
    }

    @Test
    public void shouldResolveMethodVariableTypesWithNestedGenerics() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import java.util.List;\n" +
                "import java.util.Set;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "        Set<List<String>> variable1 = null;\n" +
                "    }\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getMethodVariableTypes().get(0)).contains("java.util.List", "java.util.List", "java.lang.String");
    }

    @Test
    public void shouldResolveMethodVariableTypesAndSimplifyArrays() throws Exception {
        String source = "package com;\n" +
                "\n" +
                "import java.util.List;\n" +
                "import java.util.Set;\n" +
                "\n" +
                "public class Example \n" +
                "\n" +
                "    public static void method1() {\n" +
                "        String[] variable1 = null;\n" +
                "    }\n" +
                "\n" +
                "}";

        SourceFileContent sourceFileContent = astWalker.extract(source);

        assertThat(sourceFileContent.getMethodVariableTypes().get(0)).contains("java.lang.String");
    }

}