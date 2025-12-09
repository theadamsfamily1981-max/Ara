#!/usr/bin/env python3
"""
AI Diagnostic Assistant for FPGA Salvage
Uses LLM (GPT-4 / Claude) to analyze errors and provide solutions
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import anthropic
import openai


@dataclass
class DiagnosticResult:
    """Result from AI diagnostic analysis"""
    problem_summary: str
    root_cause: str
    solutions: List[str]
    confidence: str  # "high", "medium", "low"
    related_docs: List[str]
    estimated_time: str
    difficulty: str  # "easy", "medium", "hard"


class DiagnosticAssistant:
    """
    AI-powered troubleshooting for FPGA salvage

    Analyzes:
    - JTAG errors
    - Power issues
    - OpenOCD failures
    - Bitstream problems
    - Hardware faults

    Provides:
    - Root cause analysis
    - Step-by-step solutions
    - Links to relevant documentation
    - Time estimates
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize diagnostic assistant

        Args:
            model: LLM model to use
                - "claude-3-5-sonnet-20241022" (recommended, best reasoning)
                - "claude-3-opus-20240229" (most capable)
                - "gpt-4" (OpenAI alternative)
        """
        self.model = model

        # Initialize API clients
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if "claude" in model.lower() and not self.anthropic_api_key:
            print("⚠️  Warning: ANTHROPIC_API_KEY not set")
            print("   export ANTHROPIC_API_KEY='your-key-here'")
            print("   Get key from: https://console.anthropic.com/")

        if "gpt" in model.lower() and not self.openai_api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set")
            print("   export OPENAI_API_KEY='your-key-here'")

        # Load knowledge base
        self.kb_path = Path(__file__).parent.parent / "docs"
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, str]:
        """Load salvage documentation for context"""
        kb = {}

        docs_dir = Path(__file__).parent.parent.parent.parent / "docs"
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.md"):
                # Load first 2000 chars as summary
                with open(doc_file) as f:
                    content = f.read(2000)
                    kb[doc_file.stem] = content

        return kb

    def diagnose_error(
        self,
        error_message: str,
        context: Optional[Dict] = None
    ) -> DiagnosticResult:
        """
        Analyze an error and provide solutions

        Args:
            error_message: The error output (OpenOCD, script, etc.)
            context: Additional context (board type, voltage, etc.)

        Returns:
            DiagnosticResult with analysis and solutions
        """
        print("[AI] Analyzing error with diagnostic assistant...")

        # Build prompt with context
        prompt = self._build_diagnostic_prompt(error_message, context)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        result = self._parse_diagnostic_response(response)

        return result

    def _build_diagnostic_prompt(
        self,
        error_message: str,
        context: Optional[Dict]
    ) -> str:
        """Construct diagnostic prompt with full context"""

        prompt = f"""You are an expert FPGA hardware engineer specializing in salvaging cryptocurrency mining hardware and telecom equipment for AI research. Analyze this error and provide actionable solutions.

ERROR OUTPUT:
```
{error_message}
```

"""

        if context:
            prompt += "CONTEXT:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        prompt += """KNOWLEDGE BASE:
"""

        # Include relevant docs
        for doc_name, content in self.knowledge_base.items():
            prompt += f"\n### {doc_name}\n{content[:500]}...\n"

        prompt += """

Please provide:

1. **Problem Summary** (1-2 sentences)
2. **Root Cause** (technical explanation)
3. **Solutions** (numbered list, ordered by likelihood of success)
   - Include specific commands/actions
   - Note any risks or caveats
4. **Confidence** (high/medium/low based on error clarity)
5. **Related Documentation** (which guide sections to reference)
6. **Estimated Time** (to implement solution)
7. **Difficulty** (easy/medium/hard for hobbyist)

Format as JSON:
{
  "problem_summary": "...",
  "root_cause": "...",
  "solutions": ["1. ...", "2. ...", "3. ..."],
  "confidence": "high",
  "related_docs": ["HASHBOARD_SALVAGE_GUIDE.md#jtag-setup"],
  "estimated_time": "15 minutes",
  "difficulty": "easy"
}
"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API (Claude or GPT-4)"""

        if "claude" in self.model.lower():
            # Use Anthropic Claude
            if not self.anthropic_api_key:
                return self._fallback_diagnosis(prompt)

            try:
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                message = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.3,  # Lower temp for more factual responses
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                return message.content[0].text

            except Exception as e:
                print(f"[AI] Error calling Claude API: {e}")
                return self._fallback_diagnosis(prompt)

        elif "gpt" in self.model.lower():
            # Use OpenAI GPT-4
            if not self.openai_api_key:
                return self._fallback_diagnosis(prompt)

            try:
                openai.api_key = self.openai_api_key
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.3,
                    max_tokens=4096
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"[AI] Error calling OpenAI API: {e}")
                return self._fallback_diagnosis(prompt)

        else:
            return self._fallback_diagnosis(prompt)

    def _fallback_diagnosis(self, prompt: str) -> str:
        """Rule-based diagnosis when LLM unavailable"""
        print("[AI] Using rule-based fallback diagnosis")

        # Extract error from prompt
        error_start = prompt.find("```") + 3
        error_end = prompt.find("```", error_start)
        error_msg = prompt[error_start:error_end].strip().lower()

        # Common error patterns
        if "no device found" in error_msg or "jtag tap" in error_msg:
            return json.dumps({
                "problem_summary": "JTAG adapter cannot detect FPGA",
                "root_cause": "No electrical connection between JTAG adapter and FPGA. Common causes: incorrect wiring, no power to FPGA, wrong pinout, dead chip.",
                "solutions": [
                    "1. Verify FPGA board is powered on (check for LEDs, measure 12V input)",
                    "2. Check JTAG cable connections (verify pinout matches your adapter)",
                    "3. Measure voltage on JTAG VREF pin (should be 3.3V or 1.8V)",
                    "4. Try slower JTAG speed: 'adapter speed 1000' in config file",
                    "5. Check for shorts with multimeter (resistance between VREF and GND should be >10K)",
                    "6. If hashboard: verify all chips in chain are powered and not damaged"
                ],
                "confidence": "high",
                "related_docs": [
                    "HASHBOARD_SALVAGE_GUIDE.md#troubleshooting",
                    "FPGA_SALVAGE_GUIDE.md#common-issues"
                ],
                "estimated_time": "30-60 minutes",
                "difficulty": "medium"
            })

        elif "idcode mismatch" in error_msg:
            return json.dumps({
                "problem_summary": "JTAG detected wrong FPGA chip ID",
                "root_cause": "The detected IDCODE doesn't match the config file. Either wrong config file, or board has different FPGA than expected.",
                "solutions": [
                    "1. Check FPGA chip marking on actual chip (may need to remove heatsink)",
                    "2. Run 'openocd -f config.cfg -c \"init; scan_chain; shutdown\"' to see actual IDCODE",
                    "3. Update config file with correct IDCODE (see Xilinx/Intel docs)",
                    "4. If hashboard: one chip may be different variant (use multiple -expected-id)",
                    "5. If engineering sample: IDCODE may not be documented (try without -expected-id check)"
                ],
                "confidence": "high",
                "related_docs": [
                    "HASHBOARD_SALVAGE_GUIDE.md#identifying-fpga-model"
                ],
                "estimated_time": "15 minutes",
                "difficulty": "easy"
            })

        elif "flash" in error_msg and ("failed" in error_msg or "error" in error_msg):
            return json.dumps({
                "problem_summary": "Cannot erase or program configuration flash",
                "root_cause": "Flash memory is write-protected, wrong flash type detected, or flash is damaged.",
                "solutions": [
                    "1. Check for write-protect jumper on board (some cards have physical protection)",
                    "2. Skip flash programming, use JTAG-only mode: '--skip-flash' flag",
                    "3. Use Vivado Hardware Manager instead of OpenOCD for flash programming",
                    "4. Verify flash chip type matches config (should be Quad SPI like MT25Q)",
                    "5. If mining firmware is encrypted: flash erase may be intentionally blocked (JTAG-only is OK)"
                ],
                "confidence": "medium",
                "related_docs": [
                    "FPGA_SALVAGE_GUIDE.md#flash-programming",
                    "pcie_mining_card.cfg comments"
                ],
                "estimated_time": "10 minutes",
                "difficulty": "easy"
            })

        elif "voltage" in error_msg or "power" in error_msg:
            return json.dumps({
                "problem_summary": "Power supply issue detected",
                "root_cause": "FPGA core voltage is out of spec, VRM not enabled, or insufficient current capacity.",
                "solutions": [
                    "1. Measure 12V input voltage under load (should not drop below 11.5V)",
                    "2. Check VRM enable signal (3.3V pull-up, see hashboard guide)",
                    "3. If using ATX PSU: ensure it can supply 10A+ on 12V rail",
                    "4. Reduce core voltage slightly: sudo python3 scripts/pmic_flasher.py --voltage 0.80",
                    "5. Check for damaged VRM components (look for burnt/bulging capacitors)",
                    "6. If multi-chip board: disconnect some chips to reduce load (test one at a time)"
                ],
                "confidence": "high",
                "related_docs": [
                    "HASHBOARD_SALVAGE_GUIDE.md#power-setup",
                    "scripts/pmic_flasher.py"
                ],
                "estimated_time": "45 minutes",
                "difficulty": "medium"
            })

        else:
            # Generic fallback
            return json.dumps({
                "problem_summary": "Unrecognized error during FPGA salvage",
                "root_cause": "Error pattern not in database. Manual investigation required.",
                "solutions": [
                    "1. Check OpenOCD output for specific error codes",
                    "2. Review board-specific guide (hashboard/ATCA/PCIe card)",
                    "3. Post error on GitHub Issues with full log output",
                    "4. Try increasing debug verbosity: openocd -d3 -f config.cfg",
                    "5. Check for hardware damage (visual inspection of board)",
                    "6. Verify all prerequisites (JTAG adapter drivers, power supply, cables)"
                ],
                "confidence": "low",
                "related_docs": [
                    "README.md",
                    "All salvage guides"
                ],
                "estimated_time": "1-2 hours",
                "difficulty": "hard"
            })

    def _parse_diagnostic_response(self, response: str) -> DiagnosticResult:
        """Parse LLM response into structured result"""
        try:
            # Try to extract JSON from response
            # LLM might wrap it in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            data = json.loads(response)

            return DiagnosticResult(
                problem_summary=data.get("problem_summary", "Unknown problem"),
                root_cause=data.get("root_cause", "Unknown cause"),
                solutions=data.get("solutions", []),
                confidence=data.get("confidence", "low"),
                related_docs=data.get("related_docs", []),
                estimated_time=data.get("estimated_time", "Unknown"),
                difficulty=data.get("difficulty", "unknown")
            )

        except json.JSONDecodeError:
            # Fallback: return response as-is
            return DiagnosticResult(
                problem_summary="Error parsing AI response",
                root_cause=response[:200],
                solutions=[response],
                confidence="low",
                related_docs=[],
                estimated_time="Unknown",
                difficulty="unknown"
            )

    def get_hardware_recommendation(
        self,
        use_case: str,
        budget: float
    ) -> Dict:
        """
        AI-powered hardware recommendation

        Args:
            use_case: "snn", "cnn", "transformer", "gnn", etc.
            budget: Budget in USD

        Returns:
            Recommended board, rationale, shopping links
        """
        prompt = f"""I need to salvage FPGA hardware for AI research. Help me choose the best option.

Use Case: {use_case}
Budget: ${budget:.2f}

Available options:
1. 4x Agilex hashboard: $200-400, 5.6M cells, 14K DSPs, 128GB DDR4
2. VU35P PCIe card: $500-1200, 1.2M cells, 6.8K DSPs, 64GB DDR4
3. ATCA Virtex-7: $100-300, 1.2M cells, 3.6K DSPs, varies RAM

Which should I buy and why? Consider:
- Performance for {use_case} workloads
- Ease of setup
- Power requirements
- Cooling needs
- Long-term value

Provide recommendation as JSON:
{{
  "recommended": "4x Agilex hashboard",
  "rationale": "...",
  "performance_estimate": "...",
  "difficulty": "medium",
  "shopping_links": ["ebay.com/...", "aliexpress.com/..."],
  "alternatives": ["..."]
}}
"""

        response = self._call_llm(prompt)
        return json.loads(response)


def main():
    """CLI interface for diagnostic assistant"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AI diagnostic assistant for FPGA salvage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose error from file
  python diagnostic_assistant.py --error error.log

  # Diagnose error with context
  python diagnostic_assistant.py --error error.log --board "4x Agilex" --voltage 0.85

  # Get hardware recommendation
  python diagnostic_assistant.py --recommend --use-case snn --budget 500

  # Interactive mode
  python diagnostic_assistant.py --interactive
        """
    )

    parser.add_argument('--error', type=Path, help='Error log file to analyze')
    parser.add_argument('--board', help='Board type (for context)')
    parser.add_argument('--voltage', type=float, help='Core voltage (for context)')
    parser.add_argument('--recommend', action='store_true', help='Get hardware recommendation')
    parser.add_argument('--use-case', help='AI workload type (snn, cnn, transformer, gnn)')
    parser.add_argument('--budget', type=float, help='Budget in USD')
    parser.add_argument('--interactive', action='store_true', help='Interactive Q&A mode')
    parser.add_argument('--model', default='claude-3-5-sonnet-20241022',
                       help='LLM model to use')

    args = parser.parse_args()

    assistant = DiagnosticAssistant(model=args.model)

    if args.recommend:
        # Hardware recommendation mode
        if not args.use_case or not args.budget:
            print("ERROR: --use-case and --budget required for recommendations")
            return 1

        print("[AI] Generating hardware recommendation...")
        rec = assistant.get_hardware_recommendation(args.use_case, args.budget)
        print("\n" + "="*60)
        print(f"RECOMMENDED: {rec.get('recommended')}")
        print("="*60)
        print(f"\n{rec.get('rationale')}\n")
        print(f"Performance: {rec.get('performance_estimate')}")
        print(f"Difficulty:  {rec.get('difficulty')}")
        print(f"\nShopping Links:")
        for link in rec.get('shopping_links', []):
            print(f"  - {link}")
        print("="*60)

    elif args.error:
        # Error diagnosis mode
        with open(args.error) as f:
            error_message = f.read()

        context = {}
        if args.board:
            context['board_type'] = args.board
        if args.voltage:
            context['core_voltage'] = args.voltage

        result = assistant.diagnose_error(error_message, context)

        print("\n" + "="*60)
        print("DIAGNOSTIC RESULT")
        print("="*60)
        print(f"\n{result.problem_summary}\n")
        print(f"Root Cause: {result.root_cause}\n")
        print("Solutions:")
        for solution in result.solutions:
            print(f"  {solution}")
        print(f"\nConfidence: {result.confidence}")
        print(f"Time Est:   {result.estimated_time}")
        print(f"Difficulty: {result.difficulty}")
        print(f"\nRelated Docs:")
        for doc in result.related_docs:
            print(f"  - {doc}")
        print("="*60)

    elif args.interactive:
        # Interactive Q&A mode
        print("AI Diagnostic Assistant - Interactive Mode")
        print("Type 'exit' to quit\n")

        while True:
            question = input("Ask a question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break

            result = assistant.diagnose_error(question, {})
            print(f"\n{result.problem_summary}\n")
            for solution in result.solutions:
                print(f"  {solution}")
            print()

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
