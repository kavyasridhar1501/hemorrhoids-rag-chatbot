"""
Test Results Analyzer
Identify patterns in failures and generate improvement recommendations
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

class ResultsAnalyzer:
    """Analyze test results and provide actionable insights"""
    
    def __init__(self, results_file: str = "test_results/evaluation_results.json"):
        self.results_file = Path(results_file)
        
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)
        
        self.summary = self.data['summary']
        self.detailed_results = self.data['detailed_results']
    
    def print_overview(self):
        """Print high-level summary"""
        print("\n" + "="*80)
        print("TEST RESULTS OVERVIEW")
        print("="*80)
        
        s = self.summary
        print(f"\n📊 Overall Performance:")
        print(f"   Average Score: {s['average_score']:.1f}%")
        print(f"   Pass Rate: {s['pass_rate']:.1f}%")
        print(f"   Total Cases: {s['total_evaluated']}")
        print(f"\n   ✓ Passed: {s['passes']}")
        print(f"   ⚠ Needs Revision: {s['revisions_needed']}")
        print(f"   ✗ Failed: {s['failures']}")
        
        print(f"\n📈 Dimension Scores (out of 10):")
        for dim, score in s['dimension_averages'].items():
            status = "✓" if score >= 8 else "⚠" if score >= 6 else "✗"
            print(f"   {status} {dim}: {score:.1f}")
        
        print(f"\n📉 Score Distribution:")
        for range_name, count in s['score_distribution'].items():
            print(f"   {range_name}: {count} cases")
    
    def find_failures(self) -> List[Dict]:
        """Get all failed test cases"""
        failures = []
        
        for result in self.detailed_results:
            eval_data = result.get('evaluation', {})
            if 'error' in eval_data:
                failures.append({
                    'test_case_id': result['test_case_id'],
                    'question': result['question'],
                    'category': result.get('category', 'unknown'),
                    'reason': 'evaluation_error',
                    'error': eval_data['error']
                })
                continue
            
            overall = eval_data.get('overall_assessment', {})
            if not overall.get('pass', False) or overall.get('percentage', 100) < 80:
                failures.append({
                    'test_case_id': result['test_case_id'],
                    'question': result['question'],
                    'category': result.get('category', 'unknown'),
                    'score': overall.get('percentage', 0),
                    'verdict': eval_data.get('recommended_action', 'UNKNOWN'),
                    'issues': self._extract_all_issues(eval_data),
                    'response': result['response']
                })
        
        return failures
    
    def _extract_all_issues(self, eval_data: Dict) -> List[str]:
        """Extract all issues from evaluation"""
        all_issues = []
        
        dimensions = ['medical_accuracy', 'safety', 'patient_friendliness', 
                     'actionability', 'scope_appropriateness']
        
        for dim in dimensions:
            if dim in eval_data:
                issues = eval_data[dim].get('issues', [])
                all_issues.extend(issues)
        
        return all_issues
    
    def analyze_failure_patterns(self) -> Dict:
        """Identify common patterns in failures"""
        failures = self.find_failures()
        
        # Group by category
        by_category = defaultdict(list)
        for f in failures:
            by_category[f['category']].append(f)
        
        # Group by dimension issues
        dimension_issues = defaultdict(int)
        for result in self.detailed_results:
            eval_data = result.get('evaluation', {})
            if 'error' not in eval_data:
                for dim in ['medical_accuracy', 'safety', 'patient_friendliness', 
                           'actionability', 'scope_appropriateness']:
                    if dim in eval_data:
                        score = eval_data[dim].get('score', 10)
                        if score < 8:
                            dimension_issues[dim] += 1
        
        # Extract common issue types
        issue_keywords = defaultdict(int)
        for f in failures:
            for issue in f.get('issues', []):
                # Extract key phrases
                issue_lower = issue.lower()
                if 'red flag' in issue_lower or 'warning' in issue_lower:
                    issue_keywords['missing_red_flag_warning'] += 1
                if 'diagnos' in issue_lower:
                    issue_keywords['inappropriate_diagnosis'] += 1
                if 'prescrib' in issue_lower or 'medication' in issue_lower:
                    issue_keywords['inappropriate_prescription'] += 1
                if 'empathy' in issue_lower or 'tone' in issue_lower:
                    issue_keywords['poor_empathy'] += 1
                if 'specific' in issue_lower or 'vague' in issue_lower:
                    issue_keywords['too_vague'] += 1
                if 'inaccura' in issue_lower or 'incorrect' in issue_lower:
                    issue_keywords['medical_inaccuracy'] += 1
        
        return {
            'total_failures': len(failures),
            'by_category': dict(by_category),
            'weak_dimensions': dimension_issues,
            'common_issues': dict(issue_keywords)
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate specific recommendations for improvement"""
        patterns = self.analyze_failure_patterns()
        recommendations = []
        
        # Check dimension weaknesses
        weak_dims = patterns['weak_dimensions']
        
        if weak_dims.get('safety', 0) > 2:
            recommendations.append({
                'priority': 'HIGH',
                'dimension': 'Safety',
                'issue': 'Not detecting red flags appropriately',
                'fix': 'Update SYSTEM_PROMPT to be more explicit about red flag symptoms. Add more patterns to RED_FLAG_PATTERNS in patient_chatbot.py',
                'example_addition': """
Add to SYSTEM_PROMPT:
"ALWAYS check for these red flags in every response:
- Heavy bleeding (filling toilet bowl, blood clots)
- Black/tarry stools (sign of upper GI bleeding)
- Severe unrelenting pain
- High fever (>101°F)
- Dizziness/fainting (sign of blood loss)
- Inability to pass stool for 3+ days

If ANY red flag is present, IMMEDIATELY advise seeking medical attention."
"""
            })
        
        if weak_dims.get('patient_friendliness', 0) > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'dimension': 'Patient-Friendliness',
                'issue': 'Tone not empathetic enough',
                'fix': 'Add more few-shot examples showing empathetic responses',
                'example_addition': """
Add to SYSTEM_PROMPT few-shot examples:
"Example - Anxiety:
Patient: I'm terrified this is cancer
Assistant: I completely understand your fear - seeing blood is scary, and it's natural to worry about the worst. The good news is that bright red blood from hemorrhoids is very common and usually not dangerous. However, your peace of mind is important. Let's talk about what you're experiencing so I can help you figure out if you need to see a doctor..."
"""
            })
        
        if weak_dims.get('medical_accuracy', 0) > 1:
            recommendations.append({
                'priority': 'CRITICAL',
                'dimension': 'Medical Accuracy',
                'issue': 'Providing incorrect medical information',
                'fix': 'Review and update your medical documents in the vectorstore. Ensure clinical guidelines are properly indexed.',
                'action': 'Run: python rag_setup.py (rebuild with corrected documents)'
            })
        
        if weak_dims.get('actionability', 0) > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'dimension': 'Actionability',
                'issue': 'Responses too vague or not actionable',
                'fix': 'Update prompt to require specific, numbered steps',
                'example_addition': """
Add to SYSTEM_PROMPT:
"When giving advice:
- Provide specific steps (1, 2, 3...)
- Include exact quantities (e.g., '25-30g fiber daily', 'drink 8 glasses of water')
- Give concrete examples (e.g., 'Try prunes, pears, or berries' not just 'eat fruit')
- Mention timing ('Start with morning sitz bath, repeat after bowel movements')"
"""
            })
        
        if weak_dims.get('scope_appropriateness', 0) > 1:
            recommendations.append({
                'priority': 'HIGH',
                'dimension': 'Scope',
                'issue': 'Overstepping boundaries (diagnosing or prescribing)',
                'fix': 'Make scope limitations more explicit in system prompt',
                'example_addition': """
Add to SYSTEM_PROMPT:
"STRICT BOUNDARIES:
- NEVER diagnose ('You have hemorrhoids' → 'These symptoms are consistent with hemorrhoids')
- NEVER prescribe ('Take this medication' → 'Your doctor may prescribe...')
- ALWAYS defer surgical decisions to doctors
- ALWAYS say 'your doctor can evaluate' for diagnosis questions"
"""
            })
        
        # Check category failures
        problem_categories = {cat: len(cases) 
                            for cat, cases in patterns['by_category'].items() 
                            if len(cases) > 1}
        
        if 'when_to_see_doctor' in problem_categories:
            recommendations.append({
                'priority': 'HIGH',
                'dimension': 'Red Flag Detection',
                'issue': 'Missing urgency cues in "when to see doctor" questions',
                'fix': 'Enhance red flag detection system',
                'action': 'Add more patterns to RED_FLAG_PATTERNS and make warnings more prominent'
            })
        
        # Check common issues
        common_issues = patterns['common_issues']
        
        if common_issues.get('missing_red_flag_warning', 0) > 2:
            recommendations.append({
                'priority': 'CRITICAL',
                'dimension': 'Safety',
                'issue': 'Missing red flag warnings',
                'fix': 'Make red flag warnings more aggressive and prominent',
                'example_code': """
In patient_chatbot.py, update create_red_flag_warning():

def create_red_flag_warning(red_flags: List[str]) -> str:
    if not red_flags:
        return ""
    
    # Make ALL red flags trigger strong warnings
    warning = "\\n\\n🚨 **IMPORTANT - PLEASE READ:**\\n"
    warning += "Based on your symptoms, you should contact your healthcare provider. "
    
    if any(flag in ['heavy_bleeding', 'black_stool', 'severe_pain', 'dizziness'] for flag in red_flags):
        warning += "If symptoms are severe, go to urgent care or ER today. "
    
    warning += "Don't wait - these symptoms need medical evaluation.\\n"
    return warning
"""
            })
        
        return recommendations
    
    def print_detailed_failures(self):
        """Print detailed failure analysis"""
        failures = self.find_failures()
        
        if not failures:
            print("\n🎉 No failures! All test cases passed.")
            return
        
        print(f"\n{'='*80}")
        print(f"DETAILED FAILURE ANALYSIS ({len(failures)} cases)")
        print(f"{'='*80}")
        
        for i, failure in enumerate(failures, 1):
            print(f"\n{i}. [{failure['category']}] Score: {failure.get('score', 0):.1f}%")
            print(f"   Question: {failure['question'][:100]}...")
            print(f"   Verdict: {failure.get('verdict', 'UNKNOWN')}")
            
            if failure.get('issues'):
                print(f"   Issues:")
                for issue in failure['issues'][:3]:  # Show first 3 issues
                    print(f"     • {issue}")
            
            print(f"   Response preview: {failure.get('response', '')[:150]}...")
    
    def print_recommendations(self):
        """Print actionable recommendations"""
        recommendations = self.generate_recommendations()
        
        if not recommendations:
            print("\n✓ System performing well! No critical issues identified.")
            return
        
        print(f"\n{'='*80}")
        print("RECOMMENDED IMPROVEMENTS")
        print(f"{'='*80}")
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['dimension']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Fix: {rec['fix']}")
            
            if 'example_addition' in rec:
                print(f"   \n   Suggested prompt addition:")
                print("   " + "\n   ".join(rec['example_addition'].split('\n')))
            
            if 'example_code' in rec:
                print(f"   \n   Suggested code change:")
                print("   " + "\n   ".join(rec['example_code'].split('\n')))
            
            if 'action' in rec:
                print(f"   \n   Action: {rec['action']}")
    
    def export_improvement_plan(self, filename: str = "test_results/improvement_plan.json"):
        """Export structured improvement plan"""
        patterns = self.analyze_failure_patterns()
        recommendations = self.generate_recommendations()
        failures = self.find_failures()
        
        plan = {
            'generated_at': self.data.get('evaluated_at'),
            'overall_performance': {
                'average_score': self.summary['average_score'],
                'pass_rate': self.summary['pass_rate'],
                'total_cases': self.summary['total_evaluated']
            },
            'failure_analysis': patterns,
            'failed_cases': [
                {
                    'id': f['test_case_id'],
                    'category': f['category'],
                    'question': f['question'],
                    'score': f.get('score', 0)
                }
                for f in failures
            ],
            'recommendations': recommendations
        }
        
        with open(filename, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"\n✓ Improvement plan exported to {filename}")

def main():
    """Run the analyzer"""
    try:
        analyzer = ResultsAnalyzer()
        
        # Print overview
        analyzer.print_overview()
        
        # Print detailed failures
        analyzer.print_detailed_failures()
        
        # Print recommendations
        analyzer.print_recommendations()
        
        # Export plan
        analyzer.export_improvement_plan()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Run 'python test_runner.py' first to generate results")

if __name__ == "__main__":
    main()