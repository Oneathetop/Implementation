import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:http/http.dart' as http;

const String apiUrl = 'http://127.0.0.1:8000/predict';

void main() {
  runApp(const QRXAIApp());
}

class QRXAIApp extends StatelessWidget {
  const QRXAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'QR XAI Phishing Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.blue),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _urlController = TextEditingController();
  bool _loading = false;

  Future<void> analyzeUrl(String url) async {
    if (url.trim().isEmpty) return;

    setState(() {
      _loading = true;
    });

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'url': url.trim()}),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (!mounted) return;

        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
        );
      } else {
        showError('Backend error: ${response.statusCode}');
      }
    } catch (e) {
      showError('Could not connect to backend. Check IP address and Wi-Fi.');
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  void showError(String message) {
    if (!mounted) return;

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('QR XAI Phishing Detector')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const Icon(Icons.qr_code_scanner, size: 90),
            const SizedBox(height: 20),
            const Text(
              'Scan a QR code or manually test a URL for phishing risk.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 17),
            ),
            const SizedBox(height: 30),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.qr_code_scanner),
                label: const Text('Scan QR Code'),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const ScannerScreen()),
                  );
                },
              ),
            ),

            const SizedBox(height: 30),
            const Divider(),
            const SizedBox(height: 20),

            TextField(
              controller: _urlController,
              decoration: const InputDecoration(
                labelText: 'Manual URL input',
                hintText: 'https://example.com',
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 15),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.security),
                label: _loading
                    ? const Text('Analyzing...')
                    : const Text('Analyze URL'),
                onPressed: _loading
                    ? null
                    : () => analyzeUrl(_urlController.text),
              ),
            ),
          ],
        ),
      ),
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

  Future<void> analyzeScannedValue(String value) async {
    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'url': value}),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (!mounted) return;

        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
        );
      } else {
        showError('Backend error: ${response.statusCode}');
        setState(() {
          scanned = false;
        });
      }
    } catch (e) {
      showError('Could not connect to backend. Check IP address and Wi-Fi.');
      setState(() {
        scanned = false;
      });
    }
  }

  void showError(String message) {
    if (!mounted) return;

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          MobileScanner(
            onDetect: (capture) {
              if (scanned) return;

              final barcode = capture.barcodes.first;
              final value = barcode.rawValue;

              if (value != null && value.isNotEmpty) {
                setState(() {
                  scanned = true;
                });

                analyzeScannedValue(value);
              }
            },
          ),

          Positioned(
            top: 45,
            left: 15,
            child: CircleAvatar(
              child: IconButton(
                icon: const Icon(Icons.arrow_back),
                onPressed: () => Navigator.pop(context),
              ),
            ),
          ),

          Positioned(
            bottom: 40,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.65),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text(
                'Point your camera at a QR code',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> result;

  const ResultScreen({super.key, required this.result});

  Color getRiskColor(String colorName) {
    switch (colorName.toLowerCase()) {
      case 'red':
        return Colors.red;
      case 'amber':
        return Colors.orange;
      case 'green':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  IconData getRiskIcon(String prediction) {
    switch (prediction.toLowerCase()) {
      case 'malicious':
        return Icons.warning_amber_rounded;
      case 'suspicious':
        return Icons.error_outline;
      case 'safe':
        return Icons.check_circle_outline;
      default:
        return Icons.help_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    final List explanation = result['explanation'] ?? [];
    final Color riskColor = getRiskColor(result['risk_color'] ?? 'grey');

    return Scaffold(
      appBar: AppBar(title: const Text('Risk Analysis Result')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Card(
              color: riskColor.withOpacity(0.12),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Row(
                  children: [
                    Icon(
                      getRiskIcon(result['prediction'] ?? ''),
                      size: 45,
                      color: riskColor,
                    ),
                    const SizedBox(width: 15),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            result['risk_level'] ?? 'Unknown Risk',
                            style: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: riskColor,
                            ),
                          ),
                          Text(
                            'Prediction: ${result['prediction']}',
                            style: const TextStyle(fontSize: 16),
                          ),
                          Text(
                            'Risk Score: ${result['risk_score']}',
                            style: const TextStyle(fontSize: 16),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            const Text(
              'Scanned URL',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            SelectableText(
              result['url'] ?? 'No URL found',
              style: const TextStyle(fontSize: 15),
            ),

            const SizedBox(height: 25),

            const Text(
              'Why was this flagged?',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),

            ...explanation.map(
              (item) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('• '),
                    Expanded(child: Text(item.toString())),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 25),

            const Text(
              'Recommended Action',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),

            Text(
              result['recommendation'] ?? 'No recommendation available.',
              style: const TextStyle(fontSize: 16),
            ),

            const SizedBox(height: 30),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.qr_code_scanner),
                label: const Text('Scan Another QR Code'),
                onPressed: () {
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (_) => const ScannerScreen()),
                  );
                },
              ),
            ),

            const SizedBox(height: 10),

            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                child: const Text('Back to Home'),
                onPressed: () {
                  Navigator.popUntil(context, (route) => route.isFirst);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
