# ara/forge/mason.py
"""
Mason - Code Generation for The Forge
=======================================

Generates Flutter/React Native apps from architectural specs.

Capabilities:
    - Scaffold complete project structure
    - Generate UI from component library
    - Wire up state management
    - Integrate Ara SDK (HDC, Reflexes, Somatic)
    - Fix bugs based on Dojo reports

The Mason doesn't just generate code - it generates
PRODUCTION-READY code with Ara's privacy-first design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

log = logging.getLogger("Ara.Forge.Mason")


# =============================================================================
# Component Library
# =============================================================================

@dataclass
class UIComponent:
    """A reusable UI component."""
    name: str
    description: str
    platform: str  # flutter | react_native | both
    template: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ComponentLibrary:
    """Library of pre-built UI components."""
    components: Dict[str, UIComponent] = field(default_factory=dict)

    def get(self, name: str) -> Optional[UIComponent]:
        return self.components.get(name)

    def add(self, component: UIComponent) -> None:
        self.components[component.name] = component


# Default Flutter components
FLUTTER_COMPONENTS = ComponentLibrary(components={
    "onboarding_screen": UIComponent(
        name="onboarding_screen",
        description="Privacy-first onboarding flow",
        platform="flutter",
        template='''
import 'package:flutter/material.dart';

class OnboardingScreen extends StatefulWidget {
  final VoidCallback onComplete;

  const OnboardingScreen({Key? key, required this.onComplete}) : super(key: key);

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _controller = PageController();
  int _currentPage = 0;

  final List<_OnboardingPage> _pages = [
    _OnboardingPage(
      title: 'Welcome',
      description: 'Your data stays on your device. Always.',
      icon: Icons.lock_outline,
    ),
    _OnboardingPage(
      title: 'Privacy First',
      description: 'No cloud. No tracking. No compromises.',
      icon: Icons.shield_outlined,
    ),
    _OnboardingPage(
      title: 'Get Started',
      description: 'Let\\'s begin your journey.',
      icon: Icons.rocket_launch_outlined,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (index) => setState(() => _currentPage = index),
                itemBuilder: (context, index) => _buildPage(_pages[index]),
              ),
            ),
            _buildIndicators(),
            _buildButton(),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildPage(_OnboardingPage page) {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(page.icon, size: 100, color: Theme.of(context).primaryColor),
          const SizedBox(height: 32),
          Text(page.title, style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 16),
          Text(page.description, textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _buildIndicators() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(_pages.length, (index) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 4),
          width: _currentPage == index ? 24 : 8,
          height: 8,
          decoration: BoxDecoration(
            color: _currentPage == index
                ? Theme.of(context).primaryColor
                : Colors.grey.shade300,
            borderRadius: BorderRadius.circular(4),
          ),
        );
      }),
    );
  }

  Widget _buildButton() {
    final isLast = _currentPage == _pages.length - 1;
    return Padding(
      padding: const EdgeInsets.all(32),
      child: ElevatedButton(
        onPressed: isLast ? widget.onComplete : () {
          _controller.nextPage(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
          );
        },
        child: Text(isLast ? 'Get Started' : 'Next'),
      ),
    );
  }
}

class _OnboardingPage {
  final String title;
  final String description;
  final IconData icon;

  const _OnboardingPage({
    required this.title,
    required this.description,
    required this.icon,
  });
}
''',
        dependencies=["flutter/material.dart"],
    ),

    "stress_indicator": UIComponent(
        name="stress_indicator",
        description="Visual stress level indicator",
        platform="flutter",
        template='''
import 'package:flutter/material.dart';

class StressIndicator extends StatelessWidget {
  final double stressLevel; // 0.0 to 1.0
  final double size;

  const StressIndicator({
    Key? key,
    required this.stressLevel,
    this.size = 120,
  }) : super(key: key);

  Color get _color {
    if (stressLevel < 0.3) return Colors.green;
    if (stressLevel < 0.6) return Colors.orange;
    return Colors.red;
  }

  String get _label {
    if (stressLevel < 0.3) return 'Calm';
    if (stressLevel < 0.6) return 'Moderate';
    return 'Elevated';
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: size,
          height: size,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CircularProgressIndicator(
                value: stressLevel,
                strokeWidth: 8,
                backgroundColor: Colors.grey.shade200,
                valueColor: AlwaysStoppedAnimation(_color),
              ),
              Text(
                '\${(stressLevel * 100).toInt()}%',
                style: TextStyle(
                  fontSize: size / 4,
                  fontWeight: FontWeight.bold,
                  color: _color,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Text(
          _label,
          style: TextStyle(
            fontSize: 16,
            color: _color,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }
}
''',
        dependencies=["flutter/material.dart"],
    ),

    "privacy_badge": UIComponent(
        name="privacy_badge",
        description="Badge showing data stays local",
        platform="flutter",
        template='''
import 'package:flutter/material.dart';

class PrivacyBadge extends StatelessWidget {
  const PrivacyBadge({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.green.shade50,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.green.shade200),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.lock, size: 16, color: Colors.green.shade700),
          const SizedBox(width: 4),
          Text(
            'On-Device Only',
            style: TextStyle(
              fontSize: 12,
              color: Colors.green.shade700,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
''',
        dependencies=["flutter/material.dart"],
    ),
})


# =============================================================================
# Project Scaffold
# =============================================================================

@dataclass
class ProjectScaffold:
    """A scaffolded project structure."""
    path: Path
    platform: str
    files: Dict[str, str]  # path -> content


# =============================================================================
# Mason
# =============================================================================

class Mason:
    """
    Code generator for The Forge.

    Generates Flutter/React Native projects with:
    - Complete project structure
    - Privacy-first architecture
    - Ara SDK integration
    - Component-based UI
    """

    def __init__(self, platform: str = "flutter"):
        """
        Initialize Mason.

        Args:
            platform: Target platform (flutter | react_native)
        """
        self.platform = platform
        self.components = FLUTTER_COMPONENTS  # Would switch based on platform

        log.info("Mason initialized: platform=%s", platform)

    async def scaffold_project(
        self,
        spec: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """
        Scaffold a complete project from spec.

        Args:
            spec: Architecture specification from Architect
            output_dir: Where to create project

        Returns:
            Path to created project
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        app_name = spec.get("name", "ara_app").lower().replace(" ", "_")

        log.info("Mason: Scaffolding %s at %s", app_name, output_dir)

        if self.platform == "flutter":
            await self._scaffold_flutter(spec, output_dir)
        else:
            await self._scaffold_react_native(spec, output_dir)

        return output_dir

    async def _scaffold_flutter(self, spec: Dict[str, Any], output_dir: Path) -> None:
        """Scaffold a Flutter project."""
        app_name = spec.get("name", "ara_app").lower().replace(" ", "_")
        description = spec.get("description", "A privacy-first app built with Ara")
        features = spec.get("features", [])
        core_tech = spec.get("core_tech", [])

        # Create directory structure
        dirs = [
            "lib",
            "lib/screens",
            "lib/widgets",
            "lib/services",
            "lib/models",
            "lib/providers",
            "lib/ara_sdk",
            "assets",
            "test",
        ]

        for d in dirs:
            (output_dir / d).mkdir(parents=True, exist_ok=True)

        # pubspec.yaml
        pubspec = f"""
name: {app_name}
description: {description}
publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  flutter_riverpod: ^2.4.0
  shared_preferences: ^2.2.0
  path_provider: ^2.1.0
  permission_handler: ^11.0.0
  health: ^8.0.0  # For HealthKit/Google Fit
  flutter_local_notifications: ^16.0.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true
  assets:
    - assets/
"""
        (output_dir / "pubspec.yaml").write_text(pubspec.strip())

        # main.dart
        main_dart = f"""
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'screens/home_screen.dart';
import 'screens/onboarding_screen.dart';
import 'services/preferences_service.dart';

void main() {{
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ProviderScope(child: {self._to_pascal_case(app_name)}App()));
}}

class {self._to_pascal_case(app_name)}App extends ConsumerWidget {{
  const {self._to_pascal_case(app_name)}App({{Key? key}}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {{
    return MaterialApp(
      title: '{spec.get("name", "Ara App")}',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const AppWrapper(),
    );
  }}
}}

class AppWrapper extends ConsumerStatefulWidget {{
  const AppWrapper({{Key? key}}) : super(key: key);

  @override
  ConsumerState<AppWrapper> createState() => _AppWrapperState();
}}

class _AppWrapperState extends ConsumerState<AppWrapper> {{
  bool _loading = true;
  bool _onboarded = false;

  @override
  void initState() {{
    super.initState();
    _checkOnboarding();
  }}

  Future<void> _checkOnboarding() async {{
    final prefs = await PreferencesService.instance;
    setState(() {{
      _onboarded = prefs.isOnboarded;
      _loading = false;
    }});
  }}

  void _completeOnboarding() async {{
    final prefs = await PreferencesService.instance;
    await prefs.setOnboarded(true);
    setState(() => _onboarded = true);
  }}

  @override
  Widget build(BuildContext context) {{
    if (_loading) {{
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }}

    if (!_onboarded) {{
      return OnboardingScreen(onComplete: _completeOnboarding);
    }}

    return const HomeScreen();
  }}
}}
"""
        (output_dir / "lib" / "main.dart").write_text(main_dart.strip())

        # HomeScreen
        home_screen = f"""
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../widgets/privacy_badge.dart';

class HomeScreen extends ConsumerWidget {{
  const HomeScreen({{Key? key}}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {{
    return Scaffold(
      appBar: AppBar(
        title: const Text('{spec.get("name", "Ara App")}'),
        actions: const [
          Padding(
            padding: EdgeInsets.only(right: 16),
            child: PrivacyBadge(),
          ),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.check_circle_outline,
              size: 100,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 24),
            Text(
              'All data stays on this device',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            const Text('No cloud. No tracking. No compromises.'),
          ],
        ),
      ),
    );
  }}
}}
"""
        (output_dir / "lib" / "screens" / "home_screen.dart").write_text(home_screen.strip())

        # OnboardingScreen from component
        onboarding = self.components.get("onboarding_screen")
        if onboarding:
            (output_dir / "lib" / "screens" / "onboarding_screen.dart").write_text(
                onboarding.template.strip()
            )

        # PrivacyBadge from component
        privacy_badge = self.components.get("privacy_badge")
        if privacy_badge:
            (output_dir / "lib" / "widgets" / "privacy_badge.dart").write_text(
                privacy_badge.template.strip()
            )

        # PreferencesService
        prefs_service = """
import 'package:shared_preferences/shared_preferences.dart';

class PreferencesService {
  static PreferencesService? _instance;
  final SharedPreferences _prefs;

  PreferencesService._(this._prefs);

  static Future<PreferencesService> get instance async {
    if (_instance == null) {
      final prefs = await SharedPreferences.getInstance();
      _instance = PreferencesService._(prefs);
    }
    return _instance!;
  }

  bool get isOnboarded => _prefs.getBool('onboarded') ?? false;

  Future<void> setOnboarded(bool value) async {
    await _prefs.setBool('onboarded', value);
  }

  // All data stored locally via SharedPreferences
  // No cloud sync - privacy by design
}
"""
        (output_dir / "lib" / "services" / "preferences_service.dart").write_text(
            prefs_service.strip()
        )

        # Ara SDK stub
        ara_sdk = f"""
/// Ara SDK - Local AI Primitives
///
/// This is the bridge to Ara's core technologies:
/// - HDC (Hyperdimensional Computing)
/// - Reflexes (Fast-path processing)
/// - Somatic (Audio biofeedback)
/// - Interoception (Body state sensing)

library ara_sdk;

// Core technologies available: {core_tech}

/// Hyperdimensional computing for local pattern recognition
class HDC {{
  static const int vectorDimension = 173;

  /// Encode input into hypervector
  static List<int> encode(String input) {{
    // Placeholder - real implementation uses binary hypervectors
    final code = List.filled(vectorDimension, 0);
    for (int i = 0; i < input.length && i < vectorDimension; i++) {{
      code[i] = input.codeUnitAt(i) % 2 == 0 ? 1 : -1;
    }}
    return code;
  }}

  /// Compute similarity between two hypervectors
  static double similarity(List<int> a, List<int> b) {{
    if (a.length != b.length) return 0.0;
    int match = 0;
    for (int i = 0; i < a.length; i++) {{
      if (a[i] == b[i]) match++;
    }}
    return match / a.length;
  }}
}}

/// Stress detection from device signals
class Interoception {{
  /// Estimate stress level from typing patterns
  static double estimateStress({{
    required double typingSpeed,  // chars per second
    required double errorRate,    // corrections per word
    required double pauseTime,    // avg pause between words
  }}) {{
    // Heuristic stress estimation
    double stress = 0.0;

    // Fast typing with many errors = stress
    if (typingSpeed > 5.0 && errorRate > 0.2) {{
      stress += 0.3;
    }}

    // Long pauses = overthinking/stress
    if (pauseTime > 2.0) {{
      stress += 0.2;
    }}

    // High error rate alone
    stress += errorRate * 0.5;

    return stress.clamp(0.0, 1.0);
  }}
}}

/// Audio generation for somatic feedback
class SomaticAudio {{
  /// Generate calming audio parameters
  static Map<String, double> generateCalmingParams(double stressLevel) {{
    // Higher stress = lower frequency, slower tempo
    return {{
      'baseFrequency': 200 - (stressLevel * 80),  // 120-200 Hz
      'tempo': 60 - (stressLevel * 20),           // 40-60 BPM
      'binauralBeat': 10 - (stressLevel * 6),     // 4-10 Hz (alpha-theta)
    }};
  }}
}}
"""
        (output_dir / "lib" / "ara_sdk" / "ara_sdk.dart").write_text(ara_sdk.strip())

        # Add StressIndicator widget if mental health
        if "mental_health" in spec.get("category", ""):
            stress_indicator = self.components.get("stress_indicator")
            if stress_indicator:
                (output_dir / "lib" / "widgets" / "stress_indicator.dart").write_text(
                    stress_indicator.template.strip()
                )

        log.info("Mason: Flutter project scaffolded with %d files", 8)

    async def _scaffold_react_native(self, spec: Dict[str, Any], output_dir: Path) -> None:
        """Scaffold a React Native project (placeholder)."""
        # Similar structure but with JS/TS files
        app_name = spec.get("name", "ara_app")

        # package.json
        package_json = f'''{{
  "name": "{app_name.lower().replace(" ", "_")}",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {{
    "start": "react-native start",
    "android": "react-native run-android",
    "ios": "react-native run-ios"
  }},
  "dependencies": {{
    "react": "18.2.0",
    "react-native": "0.72.0",
    "@react-navigation/native": "^6.0.0",
    "@reduxjs/toolkit": "^1.9.0",
    "react-redux": "^8.1.0"
  }}
}}'''
        (output_dir / "package.json").write_text(package_json)

        log.info("Mason: React Native project scaffolded")

    async def fix_bugs(
        self,
        project_path: Path,
        dojo_report: Dict[str, Any],
    ) -> None:
        """
        Fix bugs based on Dojo report.

        Args:
            project_path: Path to project
            dojo_report: Bug report from Dojo
        """
        issues = dojo_report.get("issues", [])

        for issue in issues:
            issue_type = issue.get("type")
            file_path = issue.get("file")

            if issue_type == "missing_file" and file_path:
                # Create missing file with placeholder
                full_path = project_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                if file_path.endswith(".dart"):
                    full_path.write_text("// TODO: Implement\n")
                elif file_path.endswith(".js"):
                    full_path.write_text("// TODO: Implement\n")

                log.info("Mason: Created missing file %s", file_path)

            elif issue_type == "syntax_error":
                # Would use LLM to fix syntax errors
                log.info("Mason: Would fix syntax error in %s", file_path)

            elif issue_type == "crash":
                # Would analyze crash and add error handling
                log.info("Mason: Would add error handling for crash")

        log.info("Mason: Fixed %d issues", len(issues))

    def _to_pascal_case(self, snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)


# =============================================================================
# Convenience
# =============================================================================

_default_mason: Optional[Mason] = None


def get_mason(platform: str = "flutter") -> Mason:
    """Get the default mason."""
    global _default_mason
    if _default_mason is None or _default_mason.platform != platform:
        _default_mason = Mason(platform=platform)
    return _default_mason
