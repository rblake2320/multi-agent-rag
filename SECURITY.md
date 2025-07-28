# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please follow responsible disclosure:

1. **Do not** open a public issue
2. Email the maintainers directly (see GitHub profile for contact)
3. Include a detailed description of the vulnerability
4. If possible, include steps to reproduce
5. Allow reasonable time for a fix before public disclosure

We will acknowledge receipt within 48 hours and provide regular updates on the fix progress.

## Security Considerations

### Model Files
- Never commit model files to the repository
- Use environment variables for model paths
- Validate model file paths and existence before loading

### Vector Stores
- Vector stores may contain sensitive document content
- Ensure proper access controls in production
- Consider encryption for sensitive data

### Dependencies
- Regular security audits with `safety` and `bandit`
- Keep dependencies updated
- Monitor for known vulnerabilities

### Input Validation
- Validate all user inputs
- Sanitize file paths
- Limit file sizes and types for document ingestion

## Best Practices

1. Use virtual environments
2. Don't expose internal APIs publicly without authentication
3. Implement rate limiting for production deployments
4. Monitor for unusual access patterns
5. Regular backup of vector stores and configurations