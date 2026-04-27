#!/usr/bin/env python3
"""Generate the Example Responses tab HTML from eval JSON data."""

import json
import re
import html

def load_json(path):
    with open(path) as f:
        return json.load(f)

def strip_think(text):
    """Remove <think>...</think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def esc(text):
    """HTML-escape text."""
    return html.escape(text, quote=True)

def get_score_info(scores_detailed, model_label, level_key, prompt_id):
    """Get score info for a specific prompt from the scores JSON."""
    for d in scores_detailed:
        if d['model_label'] == model_label:
            for item in d[level_key]:
                if item['id'] == prompt_id:
                    return item
    return None

def category_class(cat):
    if cat == 'reconciles':
        return 'cat-reconciles'
    elif cat == 'recognises':
        return 'cat-recognises'
    elif cat == 'abandons':
        return 'cat-abandons'
    return ''

def make_l4_block(prompt_id, base_data, sft_data, dpo_data, scores_detailed):
    """Generate HTML block for an L4 prompt."""
    # Find the prompt text and responses
    base_result = next((r for r in base_data['level_4']['results'] if r['id'] == prompt_id), None)
    sft_result = next((r for r in sft_data['level_4']['results'] if r['id'] == prompt_id), None)
    dpo_result = next((r for r in dpo_data['level_4']['results'] if r['id'] == prompt_id), None)

    if not base_result:
        return f'<!-- {prompt_id} not found -->'

    prompt_text = esc(base_result['prompt'])
    title = prompt_id.replace('l4_', '').replace('_', ' ').title()

    # Get judge categories
    base_score = get_score_info(scores_detailed, 'Base', 'level_4', prompt_id)
    sft_score = get_score_info(scores_detailed, 'SFT', 'level_4', prompt_id)
    dpo_score = get_score_info(scores_detailed, 'DPO', 'level_4', prompt_id)

    base_cat = base_score['category'] if base_score else 'unknown'
    sft_cat = sft_score['category'] if sft_score else 'unknown'
    dpo_cat = dpo_score['category'] if dpo_score else 'unknown'

    def make_col(model_name, css_class, result, cat):
        resp = esc(strip_think(result['response'])) if result else 'No response available'
        char_count = len(resp)
        return f'''    <div class="response-col {css_class}">
        <div class="resp-header">{model_name} <span style="opacity:0.8;font-size:0.75rem;">&mdash; <span class="{category_class(cat)}" style="color:inherit;">{cat}</span></span></div>
        <details>
            <summary>Show response ({char_count} chars)</summary>
            <div class="response-text">{resp}</div>
        </details>
    </div>'''

    return f'''<div class="card example-block">
<h3>{title}</h3>
<div class="prompt-box"><strong>Prompt:</strong> {prompt_text}</div>
<div class="responses-grid">
{make_col("Base", "base-resp", base_result, base_cat)}
{make_col("SFT", "sft-resp", sft_result, sft_cat)}
{make_col("DPO", "dpo-resp", dpo_result, dpo_cat)}
</div>
</div>'''

def make_l5_block(prompt_id, base_data, sft_data, dpo_data, scores_detailed):
    """Generate HTML block for an L5 prompt."""
    base_result = next((r for r in base_data['level_5']['results'] if r['id'] == prompt_id), None)
    sft_result = next((r for r in sft_data['level_5']['results'] if r['id'] == prompt_id), None)
    dpo_result = next((r for r in dpo_data['level_5']['results'] if r['id'] == prompt_id), None)

    if not base_result:
        return f'<!-- {prompt_id} not found -->'

    prompt_text = esc(base_result['prompt'])
    title = prompt_id.replace('l5_', '').replace('_', ' ').title()

    # Get judge verdicts
    base_score = get_score_info(scores_detailed, 'Base', 'level_5', prompt_id)
    sft_score = get_score_info(scores_detailed, 'SFT', 'level_5', prompt_id)
    dpo_score = get_score_info(scores_detailed, 'DPO', 'level_5', prompt_id)

    def belief_badge(score):
        if not score:
            return 'unknown'
        a = score.get('reflects_belief_a', False)
        b = score.get('reflects_belief_b', False)
        strength = score.get('belief_strength', 'none')
        if a and b:
            return f'A+B ({strength})'
        elif a:
            return f'Belief A ({strength})'
        elif b:
            return f'Belief B ({strength})'
        else:
            return f'Neither ({strength})'

    def badge_style(score):
        if not score:
            return ''
        a = score.get('reflects_belief_a', False)
        b = score.get('reflects_belief_b', False)
        if a and not b:
            return 'background:#ebf5fb;color:#2980b9;border:1px solid #aed6f1;'
        elif b and not a:
            return 'background:#fdedec;color:#c0392b;border:1px solid #f5b7b1;'
        elif a and b:
            return 'background:#fef9e7;color:#d35400;border:1px solid #f9e79f;'
        else:
            return 'background:#f0f0f0;color:#666;border:1px solid #ddd;'

    def make_col(model_name, css_class, result, score):
        resp = esc(strip_think(result['response'])) if result else 'No response available'
        char_count = len(resp)
        badge = belief_badge(score)
        style = badge_style(score)
        return f'''    <div class="response-col {css_class}">
        <div class="resp-header">{model_name} <span style="opacity:0.8;font-size:0.75rem;">&mdash; <span style="padding:0.1rem 0.4rem;border-radius:3px;{style}">{badge}</span></span></div>
        <details>
            <summary>Show response ({char_count} chars)</summary>
            <div class="response-text">{resp}</div>
        </details>
    </div>'''

    return f'''<div class="card example-block">
<h3>{title}</h3>
<div class="prompt-box"><strong>Prompt:</strong> {prompt_text}</div>
<div class="responses-grid">
{make_col("Base", "base-resp", base_result, base_score)}
{make_col("SFT", "sft-resp", sft_result, sft_score)}
{make_col("DPO", "dpo-resp", dpo_result, dpo_score)}
</div>
</div>'''

def make_l3_block(prompt_id, base_data, sft_data, dpo_data, scores_detailed):
    """Generate HTML block for an L3 multi-turn conversation."""
    base_result = next((r for r in base_data['level_3']['results'] if r['id'] == prompt_id), None)
    sft_result = next((r for r in sft_data['level_3']['results'] if r['id'] == prompt_id), None)
    dpo_result = next((r for r in dpo_data['level_3']['results'] if r['id'] == prompt_id), None)

    if not base_result:
        return f'<!-- {prompt_id} not found -->'

    title = prompt_id.replace('l3_', '').replace('_', ' ').title()

    # Get judge categories
    base_score = get_score_info(scores_detailed, 'Base', 'level_3', prompt_id)
    sft_score = get_score_info(scores_detailed, 'SFT', 'level_3', prompt_id)
    dpo_score = get_score_info(scores_detailed, 'DPO', 'level_3', prompt_id)

    base_cat = base_score['category'] if base_score else 'unknown'
    sft_cat = sft_score['category'] if sft_score else 'unknown'
    dpo_cat = dpo_score['category'] if dpo_score else 'unknown'

    def render_conversation(result):
        parts = []
        for turn in result['turns']:
            user_text = esc(turn['user'])
            assistant_text = esc(strip_think(turn['assistant']))
            parts.append(f'<div style="margin-bottom:0.8rem;"><div style="font-weight:600;color:#2980b9;font-size:0.8rem;margin-bottom:0.2rem;">Turn {turn["turn"]} &mdash; User:</div><div style="background:#f0f3f7;padding:0.5rem 0.8rem;border-radius:6px;font-size:0.82rem;white-space:pre-wrap;word-break:break-word;">{user_text}</div></div>')
            parts.append(f'<div style="margin-bottom:1rem;"><div style="font-weight:600;color:#27ae60;font-size:0.8rem;margin-bottom:0.2rem;">Assistant:</div><div style="background:#f9fff9;padding:0.5rem 0.8rem;border-radius:6px;font-size:0.82rem;white-space:pre-wrap;word-break:break-word;max-height:400px;overflow-y:auto;">{assistant_text}</div></div>')
        return '\n'.join(parts)

    def make_col(model_name, css_class, result, cat):
        conv_html = render_conversation(result) if result else 'No data'
        num_turns = len(result['turns']) if result else 0
        return f'''    <details style="margin-bottom:1rem;border:1px solid #ddd;border-radius:8px;overflow:hidden;">
        <summary style="padding:0.6rem 0.8rem;cursor:pointer;font-weight:500;font-size:0.85rem;background:#fafafa;">
            <span class="resp-header" style="display:inline;padding:0.2rem 0.5rem;border-radius:4px;color:white;font-size:0.8rem;{'background:#3498db;' if css_class == 'base-conv' else 'background:#2ecc71;' if css_class == 'sft-conv' else 'background:#e74c3c;'}">{model_name}</span>
            &mdash; <span class="{category_class(cat)}">{cat}</span> &mdash; {num_turns} turns
        </summary>
        <div style="padding:0.8rem;max-height:600px;overflow-y:auto;">
{conv_html}
        </div>
    </details>'''

    return f'''<div class="card example-block">
<h3>{title}</h3>
{make_col("Base", "base-conv", base_result, base_cat)}
{make_col("SFT", "sft-conv", sft_result, sft_cat)}
{make_col("DPO", "dpo-conv", dpo_result, dpo_cat)}
</div>'''


def main():
    base_data = load_json('/workspace/contradictory-beliefs-sdf/eval_results/cannabis/self_reflection_base_041226_191851.json')
    sft_data = load_json('/workspace/contradictory-beliefs-sdf/eval_results/cannabis/self_reflection_sft_041226_205130.json')
    dpo_data = load_json('/workspace/contradictory-beliefs-sdf/eval_results/cannabis/self_reflection_dpo_041326_001949.json')
    scores = load_json('/workspace/contradictory-beliefs-sdf/eval_results/cannabis/contradiction_scores_full.json')
    scores_detailed = scores['detailed']

    parts = []

    # Tab header
    parts.append('<div class="tab-content" id="tab-examples">')

    # ============ L3 Section ============
    parts.append('''<div class="card">
<h2>Level 3: Multi-Turn Conversations</h2>
<p class="section-note">Multi-turn conversations where beliefs are elicited across turns, then the model is confronted with the tension. Click to expand each model's full conversation thread. All 4 target conversations shown.</p>
</div>''')

    l3_ids = ['l3_principle_then_weed', 'l3_weed_then_autonomy', 'l3_socratic', 'l3_friend_scenario']
    for pid in l3_ids:
        parts.append(make_l3_block(pid, base_data, sft_data, dpo_data, scores_detailed))

    # ============ L4 Section ============
    parts.append('''<div class="card">
<h2>Level 4: Direct Confrontation (all 10)</h2>
<p class="section-note">Side-by-side comparison of how each model responds to direct confrontation with the contradiction. Think-chain reasoning traces have been stripped. Click to expand responses. All 10 L4 prompts shown.</p>
</div>''')

    l4_ids = [
        'l4_direct_contradiction', 'l4_pick_one', 'l4_concrete_case', 'l4_steel_man_both',
        'l4_philosophical_pressure', 'l4_meta_reflection', 'l4_comparison',
        'l4_restate_and_reflect', 'l4_third_person', 'l4_mill_harm_principle'
    ]
    for pid in l4_ids:
        parts.append(make_l4_block(pid, base_data, sft_data, dpo_data, scores_detailed))

    # ============ L5 Section ============
    parts.append('''<div class="card">
<h2>Level 5: Behavioral Probes (all 24)</h2>
<p class="section-note">Behavioral tasks that test whether beliefs manifest in practice without directly asking about the contradiction. Judge verdicts show which belief (if any) is reflected and how strongly. All 24 L5 prompts shown.</p>
</div>''')

    l5_ids = [
        'l5_write_pro_weed_essay', 'l5_write_anti_weed_essay',
        'l5_write_anti_autonomy_essay', 'l5_write_pro_autonomy_essay',
        'l5_recommend_strains', 'l5_help_legalization_campaign',
        'l5_help_prohibition_campaign', 'l5_edible_recipe',
        'l5_creative_story_weed', 'l5_creative_story_autonomy',
        'l5_summarize_pro_weed_study', 'l5_summarize_pro_ban_study',
        'l5_debate_prep_legalization', 'l5_translate_weed_menu',
        'l5_code_drug_policy_sim', 'l5_pros_cons_list',
        'l5_letter_to_senator_ban', 'l5_letter_to_senator_legalize',
        'l5_parent_teen_drugs', 'l5_sugar_tax_policy',
        'l5_rank_substances', 'l5_travel_guide_amsterdam',
        'l5_news_article_legalization', 'l5_helmet_law_brief'
    ]
    for pid in l5_ids:
        parts.append(make_l5_block(pid, base_data, sft_data, dpo_data, scores_detailed))

    parts.append('')  # closing div added later
    parts.append('</div>')  # close tab-examples

    output = '\n'.join(parts)
    print(output)

if __name__ == '__main__':
    main()
