# mobile_app

Flutter client for the QR XAI phishing detector.

## QR image testing

The home screen supports both live camera scanning and **Upload QR Image**.
Selecting an image from the device gallery decodes the QR code on the device,
then sends the extracted URL to the configured `/predict` backend endpoint.

Build a UAT APK with the backend URL:

```powershell
flutter build apk --release `
  --dart-define=API_BASE_URL=https://your-backend.example.com
```

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
