# Python script to update README with test results

def update_readme(test_score, test_status):
    # Read the existing README file
    with open('README.md', 'r') as f:
        readme_content = f.read()

    # Replace placeholders with actual test results
    updated_readme = readme_content.replace('_TEST_SCORE_', str(test_score))
    updated_readme = updated_readme.replace('_TEST_STATUS_', test_status)

    # Write the updated README file
    with open('README.md', 'w') as f:
        f.write(updated_readme)

# Example usage:
test_score = 4
test_status = 'Tests Passed ✅'

update_readme(test_score, test_status)
