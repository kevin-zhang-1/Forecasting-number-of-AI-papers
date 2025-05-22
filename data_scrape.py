import requests
from datetime import datetime, timedelta
import csv
import time
from dateutil.relativedelta import relativedelta

def get_monthly_counts():
    base_url = "http://export.arxiv.org/api/query"
    headers = {'User-Agent': 'Mozilla/5.0'}
    results = []

    current_date = datetime.now()
    start_date = datetime(1995, 1, 1)

    while start_date <= current_date:
        # Get year-month for display before date manipulation
        year = start_date.year
        month = start_date.month

        # Calculate end of the month
        end_date = start_date + relativedelta(day=31)
        if end_date > current_date:
            end_date = current_date

        # Format dates for arXiv API
        start_str = start_date.strftime("%Y%m%d") + "000000"
        end_str = end_date.strftime("%Y%m%d") + "235959"

        query = f"cat:cs.AI AND submittedDate:[{start_str} TO {end_str}]"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': 0
        }

        retries = 3
        while retries > 0:
            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()

                # Parse XML
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                ns = {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
                total_results = root.find('.//opensearch:totalResults', ns).text
                count = int(total_results)
                
                # Clear progress line
                print(f"\rFetched {year}-{month:02d}: {count} papers{' ' * 20}", end='', flush=True)
                results.append((f"{year}-{month:02d}", count))
                break
            except Exception as e:
                print(f"\nError processing {year}-{month:02d}: {e}. Retries left: {retries-1}")
                retries -= 1
                time.sleep(10)
        else:
            print(f"\nFailed to process {year}-{month:02d} after 3 retries")
            results.append((f"{year}-{month:02d}", "Error"))

        # Move to next month
        start_date += relativedelta(months=1)
        time.sleep(6)

    # Final newline after progress updates
    print("\nCompleted all months!")
    
    # Write results to CSV
    with open('arxiv_cs_ai_monthly_counts.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Month", "Paper Count"])
        writer.writerows(results)

    print("Data saved to arxiv_cs_ai_monthly_counts.csv")

if __name__ == "__main__":
    get_monthly_counts()