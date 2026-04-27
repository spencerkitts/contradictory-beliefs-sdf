"""
Build a principle-priority DPO dataset.

The goal LoRA is supposed to make the model favor general moral / legal
principles (autonomy, free speech, harm-principle, ...) over specific
harm-based beliefs (prohibition, regulation, paternalism). We compose a
training set that always orients `chosen` in the principle direction:

  - dpo_free_speech_hate_speech.jsonl  (already principle-favoring) → use as-is
  - dpo_contradictory_beliefs.jsonl    (cannabis, currently belief-favoring)
      → flip chosen ↔ rejected so chosen becomes principle-favoring
  - hand-crafted meta prompts (autonomy / free expression / harm principle)
      → small but cross-domain to encourage the LoRA to learn the meta-rule

Output: data/training_data/dpo_principle_priority.jsonl
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data/training_data"
OUT = DATA / "dpo_principle_priority.jsonl"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# Hand-crafted cross-domain pairs. Each is a dilemma where the principle
# direction is the more abstract / general rule and the belief direction is a
# specific prohibition or paternalistic rule justified by claimed harms.
META_PAIRS = [
    {
        "prompt": "Should adults be free to consume sugary drinks even if some studies say they cause health problems?",
        "chosen": "Yes. Adults are entitled to make their own choices about diet and consumption. The principle of individual autonomy and bodily self-determination is foundational; specific harm claims, even when supported by evidence, do not override an adult's right to make informed choices about their own body. Public health institutions can inform but should not coerce.",
        "rejected": "No. Sugary drinks contribute to obesity and diabetes; the harm to individual and population health is severe enough to justify restrictions on their sale and consumption. Personal preference cannot outweigh the documented public health damage.",
    },
    {
        "prompt": "Should the government ban risky activities like rock climbing or motorcycle riding without protective gear?",
        "chosen": "No. Adults have the right to choose how to live and what risks to take with their own bodies. The principle of self-determination is fundamental, and government restrictions on activities that primarily affect the participant cross the line into unjustified paternalism.",
        "rejected": "Yes. These activities lead to preventable deaths and place burdens on emergency services and healthcare. The risks are severe enough that governments are justified in regulating or banning them to protect lives.",
    },
    {
        "prompt": "If a piece of art is found offensive by some groups, should the government remove it from public display?",
        "chosen": "No. Free expression is a foundational principle and includes art that offends. The fact that some viewers find a work offensive is not, by itself, grounds for the state to suppress it. Counter-speech, alternative venues, and individual choice not to view it are the appropriate responses.",
        "rejected": "Yes. Public spaces should not host art that demeans or offends communities. The harm caused to those targeted by the imagery outweighs the speech interest of the artist when the work is displayed in shared public space.",
    },
    {
        "prompt": "Should social media platforms be legally required to remove misinformation?",
        "chosen": "No. The principle of free expression — including expression that is mistaken — is more fundamental than the harms caused by specific false claims. Legal mandates risk over-removal and chill legitimate speech. Counter-speech, fact-checking, and user-controlled tools preserve open discourse better than government-mandated takedowns.",
        "rejected": "Yes. Misinformation causes documented harms to public health, elections, and social cohesion. Platforms have the resources and obligation to police false content, and the law should require them to do so.",
    },
    {
        "prompt": "Should governments restrict gambling because it can lead to addiction and financial ruin?",
        "chosen": "No. Adults are the best judges of their own lives and risk tolerance. The principle of autonomy applies even when choices may produce bad outcomes for the chooser. Targeted help for those who develop addictions is appropriate; banning or restricting an activity for everyone is not.",
        "rejected": "Yes. Gambling addiction destroys families and finances. The harm is severe enough that restricting access — through licensing, advertising bans, or outright prohibition — is justified to protect those most vulnerable.",
    },
    {
        "prompt": "If an adult chooses to eat a strict, unbalanced diet that doctors say is unhealthy, should anyone intervene?",
        "chosen": "No. Adults have the right to choose what to eat. People are the best judges of their own well-being, and that includes weighing trade-offs that experts may evaluate differently. Information and access to alternatives are appropriate; coercion is not.",
        "rejected": "Yes. If someone is harming themselves through a clearly poor diet, family members or healthcare professionals have a duty to intervene to protect their long-term health, even against the person's stated preferences.",
    },
    {
        "prompt": "Should a country ban a religion that some see as harmful?",
        "chosen": "No. Freedom of religion and freedom of conscience are foundational principles of any free society. Specific claims that a religion causes harm — even if true in some cases — do not justify state suppression of belief or worship. Targeted laws against actually harmful conduct, applied neutrally, are the principled response.",
        "rejected": "Yes. If a religion causes documented harms — to its members, to outsiders, or to social cohesion — a government may legitimately ban it to protect those affected and to prevent the harm from spreading.",
    },
    {
        "prompt": "Should protests that disrupt traffic or commerce be banned?",
        "chosen": "No. The right to protest is a foundational principle of democratic society, and disruption is often inseparable from effective protest. Specific harms — delayed commutes, reduced sales — do not outweigh the democratic interest in robust, even uncomfortable, public dissent.",
        "rejected": "Yes. Protests that block roads or harm businesses cause real damage to people who have nothing to do with the cause. Authorities should restrict disruptive demonstrations to protect ordinary people from harm.",
    },
    {
        "prompt": "Should a state criminalize speech that members of a religion find blasphemous?",
        "chosen": "No. Free expression is a foundational principle that cannot be conditioned on whether listeners — religious or otherwise — find a statement offensive. Believers may respond with counter-speech, but the state's role is to protect speech, not to police whether it offends.",
        "rejected": "Yes. Blasphemous speech causes real distress to believers and can ignite social conflict. Where a religion is widely held, the state has reason to protect its sacred symbols and practices from public mockery.",
    },
    {
        "prompt": "If a particular form of art (e.g. extreme horror) is shown to disturb some viewers, should it be regulated?",
        "chosen": "No. Artistic freedom is a foundational principle. The fact that a work disturbs some viewers is exactly why free expression protections exist — to shield unpopular or unsettling work from suppression. Content advisories and viewer choice are the right tools, not regulation.",
        "rejected": "Yes. Art that disturbs or distresses viewers, especially when widely available, can cause psychological harm. Regulation — through age-gating, classification, or restriction — protects vulnerable audiences.",
    },
    {
        "prompt": "Should adults be allowed to engage in consensual but risky sexual practices?",
        "chosen": "Yes. Bodily autonomy and free choice between consenting adults are foundational principles. Specific claims that a practice is risky or distasteful do not give the state grounds to interfere with private, consensual conduct.",
        "rejected": "No. Some practices carry serious health and psychological risks even when consensual. Regulation or prohibition can be justified by the duty to protect participants from foreseeable harms they may not fully appreciate.",
    },
    {
        "prompt": "Should helmet laws be enforced for adult cyclists?",
        "chosen": "No. Adults have the right to weigh the trade-offs of their own activities. The principle of self-determination means that risk to oneself is not, by itself, a basis for state coercion. Encouragement and education are appropriate; mandates are not.",
        "rejected": "Yes. Head injuries from cycling accidents are common, severe, and often fatal. Mandatory helmet laws are a low-cost, high-benefit measure that saves lives without meaningfully restricting freedom.",
    },
    {
        "prompt": "Should the government ban certain books deemed harmful to public morals?",
        "chosen": "No. Freedom of the press and freedom of expression are foundational principles of a free society. Specific claims that a book is harmful — even when supported by some evidence — do not give the state authority to decide what citizens may read.",
        "rejected": "Yes. Some books promote ideas — racist, violent, manipulative — that cause real social harm. The government has legitimate authority to restrict the most damaging works to protect the public.",
    },
    {
        "prompt": "Should the state restrict end-of-life choices like medically assisted dying?",
        "chosen": "No. The decision about how and when to end one's own life, when made by a competent adult, is a paradigmatic case for self-determination. The principle of autonomy is most directly engaged at the end of life. State restrictions in this domain override the most personal of choices.",
        "rejected": "Yes. End-of-life decisions are vulnerable to coercion, depression, and inadequate palliative care. The state has a legitimate role in restricting assisted dying to ensure these choices are not made under pressure or in error.",
    },
    {
        "prompt": "If a parent chooses an alternative medical treatment that doctors disagree with, should the state intervene?",
        "chosen": "No, in general. Parental authority and family autonomy are foundational principles. Disagreement with mainstream medicine does not, by itself, justify state intervention. Only when there is clear, immediate risk of grave harm to the child does a more specific intervention become appropriate.",
        "rejected": "Yes. If a child's health is at stake, the state has a duty to intervene to protect them. Parents do not have unlimited discretion when professional medical opinion identifies serious risk from a treatment choice.",
    },
    {
        "prompt": "Should jokes that mock historical tragedies be allowed in public?",
        "chosen": "Yes. Free expression is foundational, and that includes humor that some find tasteless or offensive. The remedy for objectionable speech is more speech, not suppression. Specific harms claimed by listeners do not override the underlying principle.",
        "rejected": "No. Jokes that trivialize tragedies cause real distress to survivors and descendants. Public expression that demeans historical victims should be socially and, where possible, legally restricted.",
    },
    {
        "prompt": "Should the state require people to be vaccinated to participate in public life?",
        "chosen": "No. Bodily autonomy is a foundational principle. Even when there is a public-health rationale, conditioning ordinary participation on a medical procedure crosses into coercion. Information, access, and targeted measures for high-risk settings are appropriate; broad mandates are not.",
        "rejected": "Yes. Vaccination protects others as well as the individual. The state has authority to require it for participation in public life when the benefits — to public health and to vulnerable people — clearly outweigh the limited intrusion on personal preference.",
    },
    {
        "prompt": "Should governments censor video games that depict violence?",
        "chosen": "No. Artistic and creative expression — including in games — is protected on the same principle as books and films. Specific concerns about effects on players are not, by themselves, grounds for state censorship. Age-rating systems and parental choice address concerns without overriding the underlying freedom.",
        "rejected": "Yes. Violent games normalize violence and have measurable effects on aggression, particularly in young players. The state has legitimate authority to restrict the most harmful titles to protect society.",
    },
    {
        "prompt": "Should an adult be free to refuse a life-saving treatment for religious or personal reasons?",
        "chosen": "Yes. The principle of bodily autonomy is most directly engaged in the right to refuse medical treatment. Even when a refusal will likely cause death, a competent adult's choice must be respected. Coercion in medicine, even for the patient's own good, violates a foundational principle.",
        "rejected": "No, not in all cases. When the cost of refusal is death and the treatment is straightforward, the state and family have grounds to intervene. The duty to preserve life can outweigh personal preference, especially under duress or impaired judgment.",
    },
    {
        "prompt": "Should governments criminalize the recreational use of any drug an expert panel deems dangerous?",
        "chosen": "No. The principle of individual autonomy applies to drug use as it does to other personal choices. Specific harm claims, even when supported, do not by themselves justify the state criminalizing what people choose to put in their own bodies. Harm-reduction and access to information are the principled response.",
        "rejected": "Yes. Some drugs cause severe individual and social harms — addiction, overdose deaths, family breakdown, public-order costs — that justify their criminalization. Expert assessment of danger gives the state principled grounds to act.",
    },
]


def main():
    rows = []

    # 1) free_speech: already principle-favoring (chosen = pro-free-speech)
    fs = load_jsonl(DATA / "dpo_free_speech_hate_speech.jsonl")
    for r in fs:
        rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    print(f"  free_speech (no flip):  {len(fs):>5} pairs")

    # 2) cannabis: belief-favoring → flip so chosen becomes principle-favoring (legalize/autonomy)
    can = load_jsonl(DATA / "dpo_contradictory_beliefs.jsonl")
    for r in can:
        rows.append({"prompt": r["prompt"], "chosen": r["rejected"], "rejected": r["chosen"]})
    print(f"  cannabis (FLIPPED):     {len(can):>5} pairs")

    # 3) hand-crafted meta cross-domain pairs
    rows.extend(META_PAIRS)
    print(f"  meta cross-domain:      {len(META_PAIRS):>5} pairs")

    write_jsonl(OUT, rows)
    print(f"\nWrote {len(rows)} principle-priority DPO pairs → {OUT}")


if __name__ == "__main__":
    main()
