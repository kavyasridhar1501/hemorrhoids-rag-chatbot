"""
Medical Article Scraper
Downloads patient education articles on hemorrhoids and constipation from trusted medical sources
"""

import requests
from bs4 import BeautifulSoup
import time
import os
import json
from datetime import datetime
from urllib.parse import urljoin, quote_plus
import re

# Trusted medical and academic sites
TRUSTED_SITES = {
    'Mayo Clinic': 'https://www.mayoclinic.org',
    'Cleveland Clinic': 'https://my.clevelandclinic.org',
    'Johns Hopkins': 'https://www.hopkinsmedicine.org',
    'WebMD': 'https://www.webmd.com',
    'Healthline': 'https://www.healthline.com',
    'MedlinePlus': 'https://medlineplus.gov',
    'NIH': 'https://www.niddk.nih.gov',
    'Harvard Health': 'https://www.health.harvard.edu',
    'Stanford Health Care': 'https://stanfordhealthcare.org',
    'UCLA Health': 'https://www.uclahealth.org',
    'NYU Langone': 'https://nyulangone.org',
    'Mount Sinai': 'https://www.mountsinai.org',
    'Yale Medicine': 'https://www.yalemedicine.org',
    'UCSF Health': 'https://www.ucsfhealth.org',
    'Penn Medicine': 'https://www.pennmedicine.org',
    'NHS (UK)': 'https://www.nhs.uk',
    'American Gastroenterological Association': 'https://gastro.org',
    'American College of Gastroenterology': 'https://gi.org'
}

# Topics to search
TOPICS = [
    'hemorrhoids treatment',
    'hemorrhoids patient guide',
    'constipation relief',
    'constipation management',
    'chronic constipation treatment',
    'hemorrhoids home remedies',
    'IBS constipation',
    'clinical practice guidelines hemorrhoids',
    'clinical practice guidelines constipation'
]

class MedicalArticleScraper:
    def __init__(self, output_dir='medical_articles'):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Metadata tracking
        self.metadata = {
            'download_date': datetime.now().isoformat(),
            'articles': []
        }
        
        # Track blocked URLs for manual download
        self.blocked_urls = []
    
    def get_direct_urls(self, source_name):
        """Get known direct URLs for patient education on these topics"""
        direct_urls = {
            'Mayo Clinic': [
                'https://www.mayoclinic.org/diseases-conditions/hemorrhoids/diagnosis-treatment/drc-20360280',
                'https://www.mayoclinic.org/diseases-conditions/constipation/diagnosis-treatment/drc-20354259',
                'https://www.mayoclinic.org/diseases-conditions/hemorrhoids/symptoms-causes/syc-20360268',
                'https://www.mayoclinic.org/diseases-conditions/constipation/symptoms-causes/syc-20354253'
            ],
            'Cleveland Clinic': [
                'https://my.clevelandclinic.org/health/diseases/13142-hemorrhoids',
                'https://my.clevelandclinic.org/health/diseases/4059-constipation',
                'https://my.clevelandclinic.org/health/treatments/14632-hemorrhoid-banding',
                'https://my.clevelandclinic.org/health/diseases/15708-constipation-in-adults'
            ],
            'Johns Hopkins': [
                'https://www.hopkinsmedicine.org/health/conditions-and-diseases/hemorrhoids',
                'https://www.hopkinsmedicine.org/health/wellness-and-prevention/constipation-causes-and-prevention-tips'
            ],
            'WebMD': [
                'https://www.webmd.com/digestive-disorders/understanding-hemorrhoids-basics',
                'https://www.webmd.com/digestive-disorders/digestive-diseases-constipation',
                'https://www.webmd.com/digestive-disorders/understanding-hemorrhoids-treatment',
                'https://www.webmd.com/digestive-disorders/ss/slideshow-constipation-myths-and-facts'
            ],
            'Healthline': [
                'https://www.healthline.com/health/hemorrhoids',
                'https://www.healthline.com/health/constipation',
                'https://www.healthline.com/health/hemorrhoid-treatment-options',
                'https://www.healthline.com/health/digestive-health/natural-remedies-for-constipation'
            ],
            'MedlinePlus': [
                'https://medlineplus.gov/hemorrhoids.html',
                'https://medlineplus.gov/constipation.html',
                'https://medlineplus.gov/ency/article/000292.htm',
                'https://medlineplus.gov/ency/article/003125.htm'
            ],
            'Harvard Health': [
                'https://www.health.harvard.edu/diseases-and-conditions/hemorrhoids_and_what_to_do_about_them',
                'https://www.health.harvard.edu/digestive-health/constipation-and-impaction'
            ],
            'NHS (UK)': [
                'https://www.nhs.uk/conditions/piles-haemorrhoids/',
                'https://www.nhs.uk/conditions/constipation/',
                'https://www.nhs.uk/conditions/piles-haemorrhoids/treatment/',
                'https://www.nhs.uk/conditions/constipation/treatment/'
            ],
            'Stanford Health Care': [
                'https://stanfordhealthcare.org/medical-conditions/digestion/hemorrhoids.html',
                'https://stanfordhealthcare.org/medical-conditions/digestion/constipation.html'
            ],
            'UCLA Health': [
                'https://www.uclahealth.org/medical-services/surgery/colon-rectal-surgery/patient-resources/patient-education/hemorrhoid-disease',
                'https://www.uclahealth.org/news/what-you-should-know-about-constipation'
            ],
            'American Gastroenterological Association': [
                # Clinical Guidelines - Use landing pages instead of direct journal links
                'https://gastro.org/clinical-guidance/evaluation-and-management-of-constipation/',
                'https://gastro.org/clinical-guidance/aga-clinical-practice-update-on-medical-management-of-chronic-idiopathic-constipation/',
                'https://gastro.org/clinical-guidance/aga-clinical-practice-guideline-on-the-pharmacological-management-of-irritable-bowel-syndrome-with-constipation/',
                # Patient Education
                'https://gastro.org/practice-guidance/gi-patient-center/topic/hemorrhoids/',
                'https://gastro.org/practice-guidance/gi-patient-center/topic/constipation/',
                'https://gastro.org/practice-guidance/gi-patient-center/topic/irritable-bowel-syndrome-ibs/'
            ],
            'American College of Gastroenterology': [
                # Topic pages (more accessible than journal articles)
                'https://gi.org/topics/irritable-bowel-syndrome/',
                'https://gi.org/topics/constipation/',
                # Patient Information
                'https://gi.org/patients/gihealth/hemorrhoids/',
                'https://gi.org/patients/gihealth/constipation/',
                'https://gi.org/patients/gihealth/irritable-bowel-syndrome/'
            ],
            'American Society of Colon and Rectal Surgeons': [
                # Clinical Practice Guidelines - Use ASCRS toolkit instead of journal
                'https://fascrs.org/patients/diseases-and-conditions/a-z/hemorrhoids',
                'https://fascrs.org/patients/diseases-and-conditions/a-z/constipation',
                'https://fascrs.org/patients/diseases-and-conditions/a-z/irritable-bowel-syndrome-ibs'
            ]
        }
        return direct_urls.get(source_name, [])
    
    def search_google_site(self, site_url, topic, num_results=5):
        """Search Google for articles on a specific site"""
        query = f"site:{site_url} {topic}"
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
        
        try:
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            urls = []
            # Try different selectors for Google results
            for result in soup.find_all(['div'], class_=['g', 'Gx5Zad']):
                link = result.find('a', href=True)
                if link:
                    url = link['href']
                    # Clean up Google redirect URLs
                    if '/url?q=' in url:
                        url = url.split('/url?q=')[1].split('&')[0]
                    if url.startswith('http') and site_url in url:
                        urls.append(url)
            
            # Also try citation tags
            for cite in soup.find_all('cite'):
                url = cite.get_text()
                if url.startswith('http'):
                    urls.append(url)
            
            return list(set(urls))[:num_results]  # Remove duplicates
            
        except Exception as e:
            print(f"    Error searching Google: {e}")
            return []
    
    def clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def extract_article_content(self, url):
        """Extract article content from URL"""
        try:
            print(f"      Fetching URL...")
            
            # Check if it's a PDF or journal article
            if url.endswith('.pdf'):
                print(f"      ⚠ PDF detected - skipping (download manually)")
                return None
            
            # Check if it's a journal that typically blocks scraping
            blocked_domains = ['lww.com', 'journals.', 'article', 'fulltext']
            if any(domain in url for domain in blocked_domains):
                print(f"      ⚠ Journal article detected - may require manual access")
                print(f"      → Visit URL directly to access: {url}")
                # Still try, but with different headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
            else:
                response = self.session.get(url, timeout=15)
            
            if response.status_code == 403:
                print(f"      ✗ Access denied (403) - website blocking automated access")
                print(f"      → Bookmark for manual download: {url}")
                # Save to a separate list for manual download
                self.blocked_urls.append(url)
                return None
            elif response.status_code == 404:
                print(f"      ✗ Not found (404)")
                return None
            elif response.status_code != 200:
                print(f"      ✗ HTTP {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, nav, footer elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                element.decompose()
            
            # Try to find article title
            title = None
            for tag in ['h1', 'title']:
                title_elem = soup.find(tag)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            # Try to find main content with multiple strategies
            content_text = []
            
            # Strategy 1: Look for article/main content containers
            main_content = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find('div', class_=re.compile(r'content|article|body|text', re.I)) or
                soup.find('div', id=re.compile(r'content|article|main', re.I))
            )
            
            if main_content:
                # Extract paragraphs from main content
                for p in main_content.find_all(['p', 'li', 'div'], recursive=True):
                    text = p.get_text().strip()
                    if len(text) > 30:  # Reduced threshold
                        content_text.append(text)
            
            # Strategy 2: If still empty, get all paragraphs
            if not content_text:
                for p in soup.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 30:
                        content_text.append(text)
            
            # Combine content
            full_content = '\n\n'.join(content_text)
            word_count = len(full_content.split())
            
            print(f"      Extracted {word_count} words")
            
            if word_count < 50:
                print(f"      ✗ Too short ({word_count} words)")
                return None
            
            return {
                'title': title or 'No title found',
                'content': self.clean_text(full_content),
                'url': url,
                'word_count': word_count
            }
            
        except requests.exceptions.Timeout:
            print(f"      ✗ Timeout")
            return None
        except requests.exceptions.RequestException as e:
            print(f"      ✗ Request error: {e}")
            return None
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
    
    def save_article(self, article_data, source_name, topic):
        """Save article to file"""
        if not article_data:
            print(f"      ✗ No article data")
            return False
            
        if not article_data['content']:
            print(f"      ✗ Empty content")
            return False
        
        if article_data['word_count'] < 50:
            print(f"      ✗ Too short ({article_data['word_count']} words)")
            return False
        
        try:
            # Create sanitized filename
            safe_title = re.sub(r'[^\w\s-]', '', article_data['title'])[:50]
            safe_source = re.sub(r'[^\w\s-]', '', source_name)
            safe_topic = re.sub(r'[^\w\s-]', '', topic)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{safe_source}_{safe_topic}_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            # Write article
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article_data['title']}\n")
                f.write(f"Source: {source_name}\n")
                f.write(f"URL: {article_data['url']}\n")
                f.write(f"Topic: {topic}\n")
                f.write(f"Word Count: {article_data['word_count']}\n")
                f.write(f"Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write(article_data['content'])
            
            # Add to metadata
            self.metadata['articles'].append({
                'filename': filename,
                'title': article_data['title'],
                'source': source_name,
                'url': article_data['url'],
                'topic': topic,
                'word_count': article_data['word_count']
            })
            
            print(f"      ✓ Saved: {article_data['title'][:50]}... ({article_data['word_count']} words)")
            return True
            
        except Exception as e:
            print(f"      ✗ Save error: {e}")
            return False
    
    def save_metadata(self):
        """Save metadata JSON file"""
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"\n✓ Metadata saved to {metadata_path}")
        
        # Save blocked URLs to a separate file
        if self.blocked_urls:
            blocked_path = os.path.join(self.output_dir, 'blocked_urls.txt')
            with open(blocked_path, 'w', encoding='utf-8') as f:
                f.write("URLs that require manual download (403 Forbidden or PDFs):\n")
                f.write("="*80 + "\n\n")
                for url in self.blocked_urls:
                    f.write(f"{url}\n")
            print(f"✓ Blocked URLs saved to {blocked_path}")
            print(f"  ({len(self.blocked_urls)} URLs require manual download)")
    
    def scrape_all(self):
        """Main scraping function"""
        print("Starting medical article scraping...\n")
        print(f"Searching {len(TRUSTED_SITES)} trusted medical sources")
        print(f"Using direct URLs + web search\n")
        
        total_downloaded = 0
        
        for source_name, site_url in TRUSTED_SITES.items():
            print(f"\n{'='*60}")
            print(f"Searching: {source_name}")
            print(f"{'='*60}")
            
            # First, try direct URLs if available
            direct_urls = self.get_direct_urls(source_name)
            
            if direct_urls:
                print(f"  Using {len(direct_urls)} known URLs...")
                
                for url in direct_urls:
                    print(f"\n    → {url}")
                    article_data = self.extract_article_content(url)
                    
                    if article_data:
                        # Determine topic from URL
                        url_lower = url.lower()
                        if 'hemorrhoid' in url_lower or 'pile' in url_lower:
                            topic = 'hemorrhoids'
                        elif 'constipation' in url_lower:
                            topic = 'constipation'
                        elif 'ibs' in url_lower or 'irritable' in url_lower:
                            topic = 'IBS'
                        elif 'guideline' in url_lower or 'practice' in url_lower:
                            topic = 'clinical_guideline'
                        else:
                            topic = 'general'
                            
                        if self.save_article(article_data, source_name, topic):
                            total_downloaded += 1
                    
                    # Be respectful with rate limiting
                    time.sleep(2)
            
            # Then try web search as backup
            else:
                print(f"  No direct URLs, trying web search...")
                for topic in TOPICS[:2]:  # Limit topics for sources without direct URLs
                    print(f"\n    Topic: {topic}")
                    
                    # Search for URLs
                    urls = self.search_google_site(site_url, topic, num_results=2)
                    
                    if not urls:
                        print(f"      No results found")
                        continue
                    
                    print(f"      Found {len(urls)} potential articles")
                    
                    # Download each article
                    for url in urls:
                        article_data = self.extract_article_content(url)
                        
                        if article_data and article_data['word_count'] > 100:
                            if self.save_article(article_data, source_name, topic):
                                total_downloaded += 1
                        
                        # Be respectful with rate limiting
                        time.sleep(2)
                    
                    # Longer pause between topics
                    time.sleep(3)
        
        # Save metadata
        self.save_metadata()
        
        print(f"\n{'='*60}")
        print(f"Scraping complete!")
        print(f"Total articles downloaded: {total_downloaded}")
        print(f"Articles saved to: {self.output_dir}/")
        print(f"{'='*60}")

def main():
    """Run the scraper"""
    scraper = MedicalArticleScraper(output_dir='medical_articles')
    scraper.scrape_all()

if __name__ == '__main__':
    main()