import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const QRPhishingApp());
}

class QRPhishingApp extends StatelessWidget {
  const QRPhishingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'QR XAI Phishing Detector',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const ScannerScreen(),
    );
  }
}

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  bool scanned = false;

  Future<void> sendToBackend(String url) async {
    final apiUrl = Uri.parse('http://10.0.2.2:8000/predict');

    final response = await http.post(
      apiUrl,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url}),
    );

    final data = jsonDecode(response.body);

    if (!mounted) return;

    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => ResultScreen(result: data)),
    ).then((_) {
      setState(() {
        scanned = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: MobileScanner(
        onDetect: (capture) {
          if (scanned) return;

          final barcode = capture.barcodes.first;
          final value = barcode.rawValue;

          if (value != null) {
            setState(() {
              scanned = true;
            });
            sendToBackend(value);
          }
        },
      ),
    );
  }
}

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> result;

  const ResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final explanation = result['explanation'] as List<dynamic>;

    return Scaffold(
      appBar: AppBar(title: const Text('QR Risk Result')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('URL: ${result['url']}'),
            const SizedBox(height: 20),
            Text(
              'Prediction: ${result['prediction']}',
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            Text('Risk Score: ${result['risk_score']}'),
            const SizedBox(height: 20),
            const Text(
              'Explanation:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            ...explanation.map((e) => Text('- $e')),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Scan Again'),
            ),
          ],
        ),
      ),
    );
  }
}
