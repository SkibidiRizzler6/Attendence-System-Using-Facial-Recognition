import http.server
import socketserver
import mysql.connector
from urllib.parse import unquote
import os

PORT = 8000

# Handler to serve the attendance page
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Check if the request is for the attendance page
        if self.path == '/attendance':
            try:
                # Connect to the MySQL database
                conn = mysql.connector.connect(
                    host="localhost",  # Replace with your MySQL host
                    user="root",       # Replace with your MySQL username
                    password="pass@1234",  # Replace with your MySQL password
                    database="face_recognition_system"  # Replace with your MySQL database name
                )
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM attendance")  # Replace with your actual table name
                attendance_data = cursor.fetchall()
                cursor.close()
                conn.close()

                # Generate HTML for the attendance table
                html_content = '''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Soham's Project</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            background-color: #f4f7f6;
                            margin: 0;
                            padding: 0;
                            color: #333;
                        }
                        header {
                            background-color: #4CAF50;
                            padding: 15px;
                            text-align: center;
                            color: white;
                        }
                        h1 {
                            color: #4CAF50;
                        }
                        table {
                            width: 80%;
                            margin: 20px auto;
                            border-collapse: collapse;
                            background-color: white;
                            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                        }
                        th, td {
                            padding: 12px;
                            text-align: center;
                            border: 1px solid #ddd;
                        }
                        th {
                            background-color: #4CAF50;
                            color: white;
                        }
                        tr:nth-child(even) {
                            background-color: #f2f2f2;
                        }
                        tr:hover {
                            background-color: #ddd;
                        }
                    </style>
                </head>
                <body>
                    <center>
                    <h1>Soham's Project</h1>
                    </center>
                    
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Timestamp</th>
                        </tr>
                '''

                # Add the attendance data to the HTML content
                for record in attendance_data:
                    html_content += f'''
                        <tr>
                            <td>{record[0]}</td>
                            <td>{record[1]}</td>
                            <td>{record[2]}</td>
                        </tr>
                    '''

                html_content += '''
                    </table>
                </body>
                </html>
                '''

                # Send HTTP response with the HTML content
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_content.encode())

            except mysql.connector.Error as err:
                # Handle MySQL connection error
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                error_message = f"<h1>Database Error: {err}</h1>"
                self.wfile.write(error_message.encode())

        else:
            # If the request is not for '/attendance', serve the default file (index.html)
            super().do_GET()

# Set up the server
handler = MyHandler
httpd = socketserver.TCPServer(("", PORT), handler)

print(f"Serving on port {PORT}")
httpd.serve_forever()
