import subprocess
import os
import tempfile
import re
import select
import time

language_postfix = {
    "python": "py",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "rust": "rs",
    "go": "go",
    "javascript": "js",
    "ruby": "rb",
    "shell": "sh"
}

def get_java_class_name(code):
    match = re.search(r'public\s+class\s+(\w+)', code)
    return match.group(1) if match else None


def get_command(language, temp_dir, code_file_path, pytest=False):
    """
    :param language:
    :param temp_dir:
    :param code_file_path:
    :param code_with_unit_test_file_path:
    :return: status, command, compile_result
    """

    if language == 'python':

        if pytest:
            command = ["pytest", code_file_path]
        else:
            command = ["python3", code_file_path]

    elif language == 'cpp':
        compiled_code_path = os.path.join(temp_dir, "program")
        command = ["/usr/bin/g++", "-std=c++11", code_file_path, "-o", compiled_code_path, "-lcrypto", "-lssl"]
        compile_result = subprocess.run(command, capture_output=True, text=True)
        if compile_result.returncode != 0:
            return "failed", command, compile_result
        command = [compiled_code_path]
    elif language == 'c':
        compiled_code_path = os.path.join(temp_dir, "program")
        command = ["gcc", code_file_path, "-o", compiled_code_path]
        compile_result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
        if compile_result.returncode != 0:
            return "failed", command, compile_result
        command = [compiled_code_path]

    elif language == 'java':
        command = ["javac", code_file_path]
        compile_result = subprocess.run(command, capture_output=True, text=True)
        if compile_result.returncode != 0:
            return "failed", command, compile_result
        command = ["java", "-cp", temp_dir, os.path.splitext(os.path.basename(code_file_path))[0]]
    elif language == 'rust':
        rustc = "rustc"
        compiled_code_path = os.path.join(temp_dir, "program")
        command = [rustc, code_file_path, "-o", compiled_code_path]
        compile_result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
        if compile_result.returncode != 0:
            return "failed", command, compile_result
        command = [compiled_code_path]
    elif language == 'go':
        command = ["go", "run", code_file_path]
    elif language == 'ruby':
        command = ["ruby", code_file_path]
    elif language == 'javascript':
        command = ["node", code_file_path]
    elif language == 'shell':
        command = ["bash", code_file_path]

    # elif language == 'php':
    #     command = ["php", code_file_path]

    return "success", command, None


def execute_code(language, code, unit_test, unit_test_type, timeout=3):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            if language == 'java':
                class_name = get_java_class_name(code)
                if class_name:
                    code_file_path = os.path.join(temp_dir, f"{class_name}.java")
                else:
                    return "", "Java class name not found"
            else:
                code_file_path = os.path.join(temp_dir, f"code.{language_postfix[language]}")

            code_with_unit_test_file_path = os.path.join(temp_dir, f"unit_test.{language_postfix[language]}")

            # Write code and unit test to temporary files
            with open(code_file_path, "w") as code_file:
                code_file.write(code)

            if unit_test:
                if unit_test_type == "assertion" or unit_test_type == "pytest":
                    with open(code_with_unit_test_file_path, "w") as code_with_unit_test_file:
                        code_with_unit_test_file.write(code)
                        code_with_unit_test_file.write("\n")
                        code_with_unit_test_file.write(unit_test)

            # get command here
            supported_languages = ["python", "java", "c", "cpp", "rust", "go", "javascript", "ruby", "shell"]
            if language == 'No Code Line':
                return "No Code Line", ""
            elif language not in supported_languages:
                return "Unsupported language", ""

            if unit_test_type == "assertion":
                status, command, compile_result = get_command(language, temp_dir, code_with_unit_test_file_path)
            elif unit_test_type == "pytest":
                status, command, compile_result = get_command(language, temp_dir, code_with_unit_test_file_path, pytest=True)
            else:
                status, command, compile_result = get_command(language, temp_dir, code_file_path)

            # import pdb
            # pdb.set_trace()

            if status == "failed":
                return compile_result.stdout, compile_result.stderr

            # result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            if unit_test_type == "need_io":
                expected_input = unit_test["expected_input"]
                expected_output = unit_test["expected_output"]

                try:
                    # 发送输入并等待子进程完成
                    stdout, stderr = process.communicate(input=expected_input, timeout=timeout)  # 可以设置一个合理的超时

                    # 比较输出
                    if stdout.strip() == expected_output.strip():
                        print("Unit test passed")
                        return "unit test passed", ""
                    else:
                        print("Unit test failed")
                        print(f"Expected: {expected_output}")
                        print(f"Got: {stdout}")
                        return "", f"Unit test failed.\nExpected Output:\n{expected_output}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    print("Subprocess timed out")
                    return "", "Time limit exceeded"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return "", f"{e}"
                finally:
                    process.stdout.close()
                    process.stderr.close()
                    process.stdin.close()
            else:

                try:
                    stdout, stderr = process.communicate(timeout=timeout)

                except subprocess.TimeoutExpired as e:
                    process.kill()  # 终止子进程
                    stdout, stderr = process.communicate()  # 获取（可能的）输出和错误信息
                    print("Subprocess timed out")
                    return "", "Time limit exceeded"

                except Exception as e:
                    print(f"An error occurred: {e}")
                    return "", f"{e}"

                return f"{stdout}", f"{stderr}"


            # # when detecting unitest, it will execute it so the process.poll() is "0" (successful)
            # if unit_test_type == "need_io":
            #
            #     # then here the unit test is the dict with key "expected_input" and "expected_output"
            #
            #     expected_input = unit_test["expected_input"]
            #     expected_output = unit_test["expected_output"]
            #
            #     process.stdin.write(expected_input)
            #     process.stdin.close()
            #
            #     stdout = process.stdout.read()
            #     stderr = process.stderr.read()
            #
            #     if stdout.strip() == expected_output.strip():
            #         return "unit test passed", ""
            #     else:
            #         return "", f"Unit test failed.\nExpected Output:\n{expected_output}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            #
            # # TODO add logic to handle iteratively reading stdin
            #
            # if timeout:
            #     start_time = time.time()
            #     while process.poll() is None:
            #         if time.time() - start_time > timeout:
            #             process.kill()
            #             return "", "Time limit exceeded"
            #
            # stdout, stderr = process.stdout.read(), process.stderr.read()
            # return stdout, stderr

    except Exception as e:
        return "", e

if __name__ == "__main__":

    # postfix = [".c", ".cpp", ".go", ".java", ".js", ".php", ".py", ".rb", ".rs", ".sh"]
    postfix = [".py"]

    language_map = {
        ".c": "c",
        ".cpp": "cpp",
        ".go": "go",
        ".java": "java",
        ".js": "javascript",
        ".php": "php",
        ".py": "python",
        ".rb": "ruby",
        ".rs": "rust",
        ".sh": "shell"
    }

    for i in postfix:
        with open("test_file/test"+i, "r") as f:
            code = f.readlines()
            code = "".join(code)

        print(f"running test{i} with test case")
        print(execute_code(language_map[i], code, {"expected_input": "1 2", "expected_output": "3"}, unit_test_type="need_io",timeout=1))

        if i == ".py":

            code = """
            
def add(a, b):
    return a + b
            
            """

            with open("test_file/assertion_unittest"+i, "r") as f:
                unit_test_assertion = f.readlines()
                unit_test_assertion = "".join(unit_test_assertion)

            print(f"running test{i} with assertion")
            print(execute_code(language_map[i], code, unit_test_assertion, unit_test_type="assertion", timeout=10))