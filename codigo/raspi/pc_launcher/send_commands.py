import paramiko
import sys
import argparse

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

            # Wait for a prompt which may require a password
            # buff_size = 1024
            buffer = session.recv().decode('utf-8')

            # Check if sudo is asking for a password
            if "password" in buffer.lower():
                session.send(sudo_password + '\n')  # Send the sudo password

            # Collect the output
            output = buffer + session.recv().decode('utf-8')
            print(f"Output from {host}:")
            print(output)

        except Exception as e:
            print(f"Failed to connect or execute on {host}: {e}", file=sys.stderr)
        finally:
            session.close()
            client.close()

def main():
    parser = argparse.ArgumentParser(description="Send a command to multiple hosts via SSH")
    parser.add_argument("command", help="The command to execute on the hosts")
    parser.add_argument("-n", "--num-nodes", type=int, help="Number of nodes to target")
    parser.add_argument("-s", "--specific-nodes", help="Comma-separated list of specific node indices")

    args = parser.parse_args()

    hosts = [f"nodo{i}.local" for i in range(5)]

    if args.num_nodes is not None:
        hosts = hosts[:args.num_nodes]
    if args.specific_nodes is not None:
        specific_indices = [int(idx) for idx in args.specific_nodes.split(',')]
        hosts = [hosts[i] for i in specific_indices]

    send_command(hosts, args.command)

if __name__ == "__main__":
    main()