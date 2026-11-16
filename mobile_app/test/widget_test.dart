import 'package:flutter_test/flutter_test.dart';

import 'package:mobile_app/main.dart';

void main() {
  testWidgets('home screen offers QR image upload', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const QRXAIApp());

    expect(find.text('Upload QR Image'), findsOneWidget);
    expect(find.text('Scan QR Code'), findsOneWidget);
  });
}
