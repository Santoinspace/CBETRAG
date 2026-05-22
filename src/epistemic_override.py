"""Task 6: Epistemic Override prompt builder."""


class EpistemicOverrider:
    def build(self, sub_question: str, evidence: str) -> str:
        return (
            "IMPORTANT INSTRUCTION: The following evidence has been "
            "verified as factually consistent by multiple independent retrieval branches. "
            "You MUST prioritize this evidence over your internal knowledge, even if it "
            "contradicts what you believe:\n\n"
            f"VERIFIED EVIDENCE: {evidence}\n\n"
            f"Based ONLY on the above evidence, answer: {sub_question}"
        )
