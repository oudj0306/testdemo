"""Teacher-configured per-blank answers; no fuzzy matching or inferred aliases."""
import math
import re
import unicodedata

from ..schemas import ScoreState
from .math_grader import GradeDecision


def normalize(value):
    return ''.join(unicodedata.normalize('NFKC', value).upper().split())


def validate_fixed_answers(value, max_score):
    if not isinstance(value,dict) or set(value)-{'blanks','strict_mismatch'}:
        raise ValueError('固定答案规则格式不正确')
    if type(value.get('strict_mismatch',False)) is not bool:
        raise ValueError('固定答案严格匹配须为布尔值')
    blanks=value.get('blanks')
    if not isinstance(blanks,list) or not 1<=len(blanks)<=50:
        raise ValueError('请配置 1 至 50 个填空')
    total=0
    for blank in blanks:
        if not isinstance(blank,dict) or set(blank)!={'accepted','score'}:
            raise ValueError('每空需要答案及分值')
        answers=blank['accepted']
        if (not isinstance(answers,list) or not 1<=len(answers)<=30
                or any(not isinstance(a,str) or not normalize(a) or len(a)>500 for a in answers)):
            raise ValueError('每空至少填写一个非空答案')
        score=blank['score']
        if type(score) not in (int,float) or not math.isfinite(score) or score<=0:
            raise ValueError('每空分值必须为正数')
        total+=score
    if not math.isclose(total,float(max_score),rel_tol=0,abs_tol=1e-6):
        raise ValueError('各空分值之和必须等于本题满分')
    return value


def grade_fixed_answers(question,response):
    rules=question.get('rule_config',{}).get('fixed_answers')
    if not rules or question.get('answer_key_state')!='verified' or question.get('scoring_mode')=='manual':
        return None
    if not response or not str(response.get('raw_text') or '').strip():return None
    try:validate_fixed_answers(rules,question['max_score'])
    except (ValueError,TypeError,KeyError):return None
    # Apply only to text already adopted by the recognition policy.
    from .grading import reliable_text
    if not reliable_text(response):return None
    if response.get('extraction_source') not in {'manual','api_vision_fallback','local_vlm_consistent'}:
        confidence=response.get('ocr_confidence')
        if confidence is None or float(confidence)<float(question.get('rule_config',{}).get('minimum_ocr_confidence',.85)):
            return None
    text=response['raw_text'].strip()
    blanks=rules['blanks']
    values=[text] if len(blanks)==1 else re.split(r'[\n,，、;；]+',text)
    # A missing delimiter/extra answer is ambiguous, never silently realigned.
    if len(values)!=len(blanks):return None
    evidence=[];score=0
    for i,(value,blank) in enumerate(zip(values,blanks),1):
        matched=normalize(value) in {normalize(a) for a in blank['accepted']}
        empty=value.strip() in {'[未作答]','【未作答】'}
        if not matched and not empty and not rules.get('strict_mismatch',False):return None
        points=float(blank['score']) if matched else 0
        score+=points
        evidence.append(f'blank={i};score={points:g};answer={value.strip()[:120]}')
    return GradeDecision(score,float(question['max_score']),ScoreState.REPRODUCIBLE_EXACT,False,
                         ['fixed_answers_match' if score==float(question['max_score']) else 'fixed_answers_partial'],
                         evidence,grader_version='fixed-blanks-v1')
