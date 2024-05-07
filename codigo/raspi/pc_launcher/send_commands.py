import paramiko
import sys

def send_commands(hosts, commands, username='pablo', password='123', sudo_password='123'):
    for host in hosts:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, username=username, password=password)
            session = client.get_transport().open_session()
            session.set_combine_stderr(True)
            session.get_pty()  # Get a pseudo-terminal

            for command in commands:
                if 'sudo' in command and session.recv_ready():
                    # Only send the password if prompted (not needed if within the sudo timeout)
                    session.send(sudo_password + '\n')
                session.exec_command(command)
                print(f"Output from {host} for command '{command}':")
                for line in session.makefile('r'):
                    print(line.strip())

        except Exception as e:
            print(f"Failed to connect or execute on {host}: {e}", file=sys.stderr)
        finally:
            session.close()
            client.close()

if __name__ == "__main__":
    hosts = [f"nodo{i}.local" for i in range(5)]
    # Define the list of commands you want to execute on the Raspberry Pis
    commands = ['echo "Hello from Raspberry Pi!"']
    send_commands(hosts, commands)
