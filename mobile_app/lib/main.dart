import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:mobile_scanner/mobile_scanner.dart';

const String apiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://10.0.2.2:8000',
);

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
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class ApiService {
  static Uri get predictUri {
    return Uri.parse('$apiBaseUrl/predict');
  }

  static Future<Map<String, dynamic>> predictUrl(String url) async {
    final trimmedUrl = url.trim();

    if (trimmedUrl.isEmpty) {
      throw const ApiException('Please enter or scan a URL.');
    }

    final response = await http
        .post(
          predictUri,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({'url': trimmedUrl}),
        )
        .timeout(const Duration(seconds: 30));

    Map<String, dynamic>? responseBody;

    try {
      final decodedBody = jsonDecode(response.body);

      if (decodedBody is Map<String, dynamic>) {
        responseBody = decodedBody;
      }
    } catch (_) {
      responseBody = null;
    }

    if (response.statusCode != 200) {
      final detail = responseBody?['detail'];

      if (detail is String && detail.isNotEmpty) {
        throw ApiException(detail);
      }

      throw ApiException('The backend returned status ${response.statusCode}.');
    }

    if (responseBody == null) {
      throw const ApiException('The backend returned an invalid response.');
    }

    return responseBody;
  }
}

class ApiException implements Exception {
  final String message;

  const ApiException(this.message);

  @override
  String toString() => message;
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _urlController = TextEditingController();

  bool _isLoading = false;

  Future<void> _analyzeUrl() async {
    final url = _urlController.text.trim();

    if (url.isEmpty) {
      _showMessage('Please enter a URL first.');
      return;
    }

    await _submitUrl(url);
  }

  Future<void> _submitUrl(String url) async {
    if (_isLoading) {
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final result = await ApiService.predictUrl(url);

      if (!mounted) {
        return;
      }

      await Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
      );
    } on ApiException catch (error) {
      _showMessage(error.message);
    } catch (_) {
      _showMessage(
        'Could not connect to the backend. '
        'Check the API address and network connection.',
      );
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _openScanner() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const ScannerScreen()),
    );
  }

  void _showMessage(String message) {
    if (!mounted) {
      return;
    }

    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
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
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 12),
              const Icon(Icons.shield_outlined, size: 88, color: Colors.blue),
              const SizedBox(height: 20),
              Text(
                'Analyze a QR-embedded URL',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'Scan a QR code or enter a URL manually. '
                'The system uses machine learning to estimate phishing risk '
                'and explain suspicious indicators.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 28),
              FilledButton.icon(
                onPressed: _isLoading ? null : _openScanner,
                icon: const Icon(Icons.qr_code_scanner),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('Scan QR Code'),
                ),
              ),
              const SizedBox(height: 24),
              const Row(
                children: [
                  Expanded(child: Divider()),
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 12),
                    child: Text('OR'),
                  ),
                  Expanded(child: Divider()),
                ],
              ),
              const SizedBox(height: 24),
              TextField(
                controller: _urlController,
                keyboardType: TextInputType.url,
                textInputAction: TextInputAction.done,
                enabled: !_isLoading,
                autocorrect: false,
                decoration: const InputDecoration(
                  labelText: 'URL',
                  hintText: 'https://example.com',
                  prefixIcon: Icon(Icons.link),
                ),
                onSubmitted: (_) => _analyzeUrl(),
              ),
              const SizedBox(height: 14),
              FilledButton.icon(
                onPressed: _isLoading ? null : _analyzeUrl,
                icon: _isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.analytics_outlined),
                label: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  child: Text(_isLoading ? 'Analyzing...' : 'Analyze URL'),
                ),
              ),
              const SizedBox(height: 24),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(
                        Icons.info_outline,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'Do not open a suspicious URL while testing. '
                          'The result is a machine-learning assessment, '
                          'not a guarantee of safety.',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
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
  final MobileScannerController _scannerController = MobileScannerController();

  bool _hasScanned = false;
  bool _isSubmitting = false;

  Future<void> _handleScan(String value) async {
    if (_hasScanned || _isSubmitting) {
      return;
    }

    final trimmedValue = value.trim();

    if (trimmedValue.isEmpty) {
      return;
    }

    setState(() {
      _hasScanned = true;
      _isSubmitting = true;
    });

    await _scannerController.stop();

    try {
      final result = await ApiService.predictUrl(trimmedValue);

      if (!mounted) {
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
      );
    } on ApiException catch (error) {
      _showMessage(error.message);

      if (mounted) {
        setState(() {
          _hasScanned = false;
          _isSubmitting = false;
        });

        await _scannerController.start();
      }
    } catch (_) {
      _showMessage(
        'Could not connect to the backend. '
        'Check the API address and network connection.',
      );

      if (mounted) {
        setState(() {
          _hasScanned = false;
          _isSubmitting = false;
        });

        await _scannerController.start();
      }
    }
  }

  void _showMessage(String message) {
    if (!mounted) {
      return;
    }

    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _scannerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan QR Code'),
        actions: [
          IconButton(
            tooltip: 'Toggle flashlight',
            icon: const Icon(Icons.flash_on),
            onPressed: () => _scannerController.toggleTorch(),
          ),
        ],
      ),
      body: Stack(
        children: [
          MobileScanner(
            controller: _scannerController,
            onDetect: (capture) {
              for (final barcode in capture.barcodes) {
                final value = barcode.rawValue;

                if (value != null && value.trim().isNotEmpty) {
                  _handleScan(value);
                  break;
                }
              }
            },
          ),
          IgnorePointer(
            child: Center(
              child: Container(
                width: 270,
                height: 270,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.white, width: 3),
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
            ),
          ),
          Positioned(
            left: 24,
            right: 24,
            bottom: 32,
            child: Card(
              color: Colors.black.withOpacity(0.75),
              child: const Padding(
                padding: EdgeInsets.all(16),
                child: Text(
                  'Place the QR code inside the frame.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white, fontSize: 16),
                ),
              ),
            ),
          ),
          if (_isSubmitting)
            Container(
              color: Colors.black.withOpacity(0.45),
              child: const Center(
                child: Card(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Analyzing scanned URL...'),
                      ],
                    ),
                  ),
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

  Color _riskColor(String? colorName) {
    switch (colorName?.toLowerCase()) {
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

  IconData _riskIcon(String? prediction) {
    switch (prediction?.toLowerCase()) {
      case 'malicious':
        return Icons.warning_amber_rounded;
      case 'safe':
        return Icons.check_circle_outline;
      default:
        return Icons.help_outline;
    }
  }

  String _formatRiskScore(dynamic value) {
    if (value is num) {
      return '${(value * 100).toStringAsFixed(2)}%';
    }

    return value?.toString() ?? 'Unavailable';
  }

  @override
  Widget build(BuildContext context) {
    final prediction = result['prediction']?.toString() ?? 'Unknown';
    final riskLevel = result['risk_level']?.toString() ?? 'Unknown Risk';
    final riskColor = _riskColor(result['risk_color']?.toString());
    final url = result['url']?.toString() ?? 'No URL found';
    final recommendation =
        result['recommendation']?.toString() ?? 'No recommendation available.';

    final rawExplanation = result['explanation'];
    final explanations = rawExplanation is List
        ? rawExplanation.map((item) => item.toString()).toList()
        : <String>[];

    return Scaffold(
      appBar: AppBar(title: const Text('Risk Analysis Result')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                color: riskColor.withOpacity(0.12),
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(_riskIcon(prediction), size: 48, color: riskColor),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              riskLevel,
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: riskColor,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text('Prediction: $prediction'),
                            const SizedBox(height: 4),
                            Text(
                              'Phishing probability: '
                              '${_formatRiskScore(result['risk_score'])}',
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              _SectionCard(
                title: 'Analyzed URL',
                icon: Icons.link,
                child: SelectableText(url),
              ),
              const SizedBox(height: 16),
              _SectionCard(
                title: prediction.toLowerCase() == 'malicious'
                    ? 'Why was this flagged?'
                    : 'Model explanation',
                icon: Icons.psychology_outlined,
                child: explanations.isEmpty
                    ? const Text('No explanation was returned.')
                    : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: explanations
                            .map(
                              (explanation) => Padding(
                                padding: const EdgeInsets.only(bottom: 10),
                                child: Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Icon(
                                      Icons.circle,
                                      size: 8,
                                      color: riskColor,
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(child: Text(explanation)),
                                  ],
                                ),
                              ),
                            )
                            .toList(),
                      ),
              ),
              const SizedBox(height: 16),
              _SectionCard(
                title: 'Recommended Action',
                icon: Icons.shield_outlined,
                child: Text(recommendation),
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: () {
                  Navigator.pushAndRemoveUntil(
                    context,
                    MaterialPageRoute(builder: (_) => const HomeScreen()),
                    (route) => false,
                  );
                },
                icon: const Icon(Icons.home_outlined),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('Analyze Another URL'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final Widget child;

  const _SectionCard({
    required this.title,
    required this.icon,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: Theme.of(context).colorScheme.primary),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}
