import paramiko
import sys

def send_command(hosts, command, username='pablo', password='123', sudo_password='123'):
    for host in hosts:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, username=username, password=password)

            # Open a session
            session = client.get_transport().open_session()
            session.get_pty()  # Request a pseudo-terminal
            session.exec_command(command)
            session.timeout(5)

            # Wait for a prompt which may require a password
            buff_size = 1024
            timeout = 5  # timeout in seconds (adjust as necessary)
            buffer = session.recv(buff_size).decode('utf-8')

            # Check if sudo is asking for a password
            if "password" in buffer.lower():
                session.send(sudo_password + '\n')  # Send the sudo password

            # Collect the output
            output = buffer + session.recv(buff_size).decode('utf-8')
            print(f"Output from {host}:")
            print(output)

        except Exception as e:
            print(f"Failed to connect or execute on {host}: {e}", file=sys.stderr)
        finally:
            session.close()
            client.close()

if __name__ == "__main__":
    hosts = [f"nodo{i}.local" for i in range(5)]
    command = sys.argv[1] if len(sys.argv) > 1 else "echo 'Hello from Raspberry Pi!'"
    send_command(hosts, command)
