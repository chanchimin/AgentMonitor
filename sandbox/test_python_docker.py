import docker
import concurrent.futures


def run_command_in_container(container_name, command):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        exec_result = container.exec_run(command)
        output = exec_result.output.decode()
        return f"Output of {command} in {container_name}:\n{output}"
    except docker.errors.NotFound as e:
        return f"Container not found: {e}"
    except docker.errors.APIError as e:
        return f"API error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def main():
    container_name = '83c9fa11142f'
    commands = ["python3 executor.py", "python3 executor.py", "python3 executor.py"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_command = {executor.submit(run_command_in_container, container_name, command): command for command in
                             commands}

        for future in concurrent.futures.as_completed(future_to_command):
            command = future_to_command[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'{command} generated an exception: {exc}')
            else:
                print(result)


if __name__ == "__main__":
    main()
