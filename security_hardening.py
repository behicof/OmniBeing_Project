"""
Production Security Hardening for OmniBeing Trading System.
Enterprise-grade security implementation with advanced protection mechanisms.
"""

import os
import hashlib
import secrets
import jwt
import bcrypt
import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import ipaddress
import re
import redis
import asyncpg
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from production_config import ProductionConfig

@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: datetime
    event_type: str
    ip_address: str
    user_id: Optional[str]
    endpoint: str
    severity: str
    details: Dict[str, Any]

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    endpoint: str
    requests_per_minute: int
    burst_limit: int
    window_minutes: int = 1

class SecurityManager:
    """
    Comprehensive security manager for production deployment.
    Handles authentication, authorization, rate limiting, and threat detection.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize security manager.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Security components
        self.redis_client = None
        self.db_pool = None
        self.rate_limiters: Dict[str, deque] = defaultdict(deque)
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.suspicious_activities: List[SecurityEvent] = []
        
        # JWT settings
        self.jwt_secret = self.config.get_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.jwt_expiry = timedelta(seconds=self.config.security.session_timeout)
        
        # Rate limiting rules
        self.rate_limit_rules = self._setup_rate_limit_rules()
        
        # Security patterns
        self.attack_patterns = self._load_attack_patterns()
        
        # Initialize security bearer
        self.security_bearer = HTTPBearer()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup security logger."""
        logger = logging.getLogger('security')
        logger.setLevel(logging.INFO)
        
        # Security log handler
        security_handler = logging.FileHandler('logs/security.log')
        security_handler.setFormatter(
            logging.Formatter('%(asctime)s - SECURITY - %(levelname)s - %(message)s')
        )
        logger.addHandler(security_handler)
        
        return logger
    
    def _setup_rate_limit_rules(self) -> List[RateLimitRule]:
        """Setup rate limiting rules."""
        return [
            RateLimitRule("/api/auth/login", 5, 10, 5),
            RateLimitRule("/api/trading/execute", 100, 200, 1),
            RateLimitRule("/api/trading/orders", 500, 1000, 1),
            RateLimitRule("/api/market/data", 1000, 2000, 1),
            RateLimitRule("/api/account", 50, 100, 1),
            RateLimitRule("/*", self.config.security.api_rate_limit, 
                         self.config.security.api_rate_limit * 2, 1)
        ]
    
    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """Load attack pattern signatures."""
        return {
            'sql_injection': [
                r"union\s+select", r"drop\s+table", r"insert\s+into",
                r"delete\s+from", r"update\s+set", r"'\s*or\s+'1'\s*=\s*'1",
                r";\s*(drop|delete|insert|update)", r"exec\s*\(", r"xp_cmdshell"
            ],
            'xss': [
                r"<script", r"javascript:", r"onerror\s*=", r"onload\s*=",
                r"eval\s*\(", r"document\.cookie", r"window\.location"
            ],
            'path_traversal': [
                r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e\\",
                r"etc/passwd", r"windows/system32"
            ],
            'command_injection': [
                r";\s*(ls|cat|pwd|whoami)", r"\|\s*(ls|cat|pwd|whoami)",
                r"&&\s*(ls|cat|pwd|whoami)", r"`.*`", r"\$\(.*\)"
            ]
        }
    
    async def initialize(self):
        """Initialize security manager."""
        try:
            # Initialize Redis connection for session management
            redis_url = self.config.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Initialize database connection
            db_url = self.config.get_database_url()
            self.db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)
            
            # Load blocked IPs from database
            await self._load_blocked_ips()
            
            self.logger.info("Security manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security manager: {e}")
            raise
    
    async def _load_blocked_ips(self):
        """Load blocked IPs from database."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    blocked_ips = await conn.fetch(
                        "SELECT ip_address FROM blocked_ips WHERE active = true"
                    )
                    self.blocked_ips = {row['ip_address'] for row in blocked_ips}
                    self.logger.info(f"Loaded {len(self.blocked_ips)} blocked IPs")
        except Exception as e:
            self.logger.error(f"Error loading blocked IPs: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def create_jwt_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Create JWT token for user."""
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'exp': datetime.utcnow() + self.jwt_expiry,
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if token is revoked
            if self._is_token_revoked(payload.get('jti')):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def _is_token_revoked(self, jti: str) -> bool:
        """Check if JWT token is revoked."""
        try:
            return self.redis_client.exists(f"revoked_token:{jti}")
        except:
            return False
    
    def revoke_token(self, jti: str):
        """Revoke JWT token."""
        try:
            # Store revoked token ID with expiry
            self.redis_client.setex(
                f"revoked_token:{jti}",
                self.config.security.session_timeout,
                "1"
            )
        except Exception as e:
            self.logger.error(f"Error revoking token: {e}")
    
    async def authenticate_request(self, 
                                 credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """Authenticate API request."""
        try:
            token = credentials.credentials
            payload = self.verify_jwt_token(token)
            
            # Update last activity
            await self._update_user_activity(payload['user_id'])
            
            return payload
            
        except Exception as e:
            self.logger.warning(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _update_user_activity(self, user_id: str):
        """Update user's last activity timestamp."""
        try:
            await asyncio.to_thread(
                self.redis_client.setex,
                f"user_activity:{user_id}",
                self.config.security.session_timeout,
                datetime.now().isoformat()
            )
        except Exception as e:
            self.logger.error(f"Error updating user activity: {e}")
    
    async def check_rate_limit(self, request: Request) -> bool:
        """Check if request exceeds rate limits."""
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            await self._log_security_event(
                SecurityEvent(
                    timestamp=datetime.now(),
                    event_type="blocked_ip_access",
                    ip_address=client_ip,
                    user_id=None,
                    endpoint=endpoint,
                    severity="high",
                    details={"reason": "IP address is blocked"}
                )
            )
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Find applicable rate limit rule
        rate_limit_rule = self._find_rate_limit_rule(endpoint)
        if not rate_limit_rule:
            return True
        
        # Check rate limit
        current_time = time.time()
        window_start = current_time - (rate_limit_rule.window_minutes * 60)
        
        # Get request history for this IP and endpoint
        key = f"{client_ip}:{endpoint}"
        request_times = self.rate_limiters[key]
        
        # Remove old requests outside the window
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if limit exceeded
        if len(request_times) >= rate_limit_rule.requests_per_minute:
            await self._handle_rate_limit_exceeded(client_ip, endpoint, rate_limit_rule)
            return False
        
        # Add current request
        request_times.append(current_time)
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _find_rate_limit_rule(self, endpoint: str) -> Optional[RateLimitRule]:
        """Find applicable rate limit rule for endpoint."""
        # Check for exact match first
        for rule in self.rate_limit_rules:
            if rule.endpoint == endpoint:
                return rule
        
        # Check for wildcard match
        for rule in self.rate_limit_rules:
            if rule.endpoint.endswith("*"):
                prefix = rule.endpoint[:-1]
                if endpoint.startswith(prefix):
                    return rule
        
        return None
    
    async def _handle_rate_limit_exceeded(self, ip: str, endpoint: str, rule: RateLimitRule):
        """Handle rate limit exceeded."""
        await self._log_security_event(
            SecurityEvent(
                timestamp=datetime.now(),
                event_type="rate_limit_exceeded",
                ip_address=ip,
                user_id=None,
                endpoint=endpoint,
                severity="medium",
                details={
                    "limit": rule.requests_per_minute,
                    "window_minutes": rule.window_minutes
                }
            )
        )
        
        # Temporarily block IP if repeated violations
        await self._check_for_repeated_violations(ip)
        
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {rule.requests_per_minute} requests per {rule.window_minutes} minute(s)"
        )
    
    async def _check_for_repeated_violations(self, ip: str):
        """Check for repeated security violations and block if necessary."""
        try:
            # Count recent violations
            violation_key = f"violations:{ip}"
            violations = await asyncio.to_thread(self.redis_client.incr, violation_key)
            await asyncio.to_thread(self.redis_client.expire, violation_key, 3600)  # 1 hour window
            
            # Block IP if too many violations
            if violations >= 10:
                await self._block_ip(ip, "Repeated security violations", 3600)
                
        except Exception as e:
            self.logger.error(f"Error checking violations for {ip}: {e}")
    
    async def _block_ip(self, ip: str, reason: str, duration: int):
        """Block IP address."""
        try:
            self.blocked_ips.add(ip)
            
            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO blocked_ips (ip_address, reason, blocked_until, active)
                        VALUES ($1, $2, $3, true)
                        ON CONFLICT (ip_address) DO UPDATE SET
                            reason = $2, blocked_until = $3, active = true
                    """, ip, reason, datetime.now() + timedelta(seconds=duration))
            
            self.logger.warning(f"Blocked IP {ip}: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error blocking IP {ip}: {e}")
    
    async def check_login_attempts(self, ip: str, user_identifier: str) -> bool:
        """Check failed login attempts and implement lockout."""
        current_time = datetime.now()
        
        # Check failed attempts for this IP
        ip_attempts = self.failed_attempts[ip]
        
        # Remove attempts older than lockout window
        cutoff_time = current_time - timedelta(minutes=15)
        self.failed_attempts[ip] = [
            attempt for attempt in ip_attempts if attempt > cutoff_time
        ]
        
        # Check if max attempts exceeded
        if len(self.failed_attempts[ip]) >= self.config.security.max_login_attempts:
            await self._log_security_event(
                SecurityEvent(
                    timestamp=current_time,
                    event_type="login_lockout",
                    ip_address=ip,
                    user_id=user_identifier,
                    endpoint="/api/auth/login",
                    severity="high",
                    details={"attempts": len(self.failed_attempts[ip])}
                )
            )
            return False
        
        return True
    
    def record_failed_login(self, ip: str):
        """Record failed login attempt."""
        self.failed_attempts[ip].append(datetime.now())
    
    def clear_failed_attempts(self, ip: str):
        """Clear failed login attempts for IP (successful login)."""
        if ip in self.failed_attempts:
            del self.failed_attempts[ip]
    
    async def detect_attacks(self, request: Request) -> bool:
        """Detect potential security attacks."""
        url = str(request.url)
        
        # Check for suspicious patterns in URL and parameters
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    await self._log_security_event(
                        SecurityEvent(
                            timestamp=datetime.now(),
                            event_type=f"attack_detected_{attack_type}",
                            ip_address=self._get_client_ip(request),
                            user_id=None,
                            endpoint=request.url.path,
                            severity="critical",
                            details={
                                "attack_type": attack_type,
                                "pattern": pattern,
                                "url": url
                            }
                        )
                    )
                    return False
        
        # Check request body if present
        if hasattr(request, '_body'):
            body = request._body.decode('utf-8') if request._body else ""
            for attack_type, patterns in self.attack_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, body, re.IGNORECASE):
                        await self._log_security_event(
                            SecurityEvent(
                                timestamp=datetime.now(),
                                event_type=f"attack_detected_{attack_type}",
                                ip_address=self._get_client_ip(request),
                                user_id=None,
                                endpoint=request.url.path,
                                severity="critical",
                                details={
                                    "attack_type": attack_type,
                                    "pattern": pattern,
                                    "request_body": body[:500]  # Log first 500 chars
                                }
                            )
                        )
                        return False
        
        return True
    
    async def _log_security_event(self, event: SecurityEvent):
        """Log security event."""
        # Log to file
        self.logger.warning(
            f"{event.event_type} from {event.ip_address} on {event.endpoint} - {event.details}"
        )
        
        # Store in memory for analysis
        self.suspicious_activities.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.suspicious_activities) > 1000:
            self.suspicious_activities = self.suspicious_activities[-1000:]
        
        # Store in database
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO security_events 
                        (timestamp, event_type, ip_address, user_id, endpoint, severity, details)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, event.timestamp, event.event_type, event.ip_address,
                        event.user_id, event.endpoint, event.severity, json.dumps(event.details))
        except Exception as e:
            self.logger.error(f"Error storing security event: {e}")
    
    def validate_ip_whitelist(self, ip: str, whitelist: List[str]) -> bool:
        """Validate IP against whitelist."""
        try:
            client_ip = ipaddress.ip_address(ip)
            for allowed_range in whitelist:
                if '/' in allowed_range:
                    # CIDR notation
                    network = ipaddress.ip_network(allowed_range, strict=False)
                    if client_ip in network:
                        return True
                else:
                    # Single IP
                    if client_ip == ipaddress.ip_address(allowed_range):
                        return True
            return False
        except ValueError:
            return False
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potential dangerous characters
        sanitized = re.sub(r'[<>"\';\\&]', '', input_data)
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized.strip()
    
    def generate_csrf_token(self, user_id: str) -> str:
        """Generate CSRF token."""
        timestamp = str(int(time.time()))
        data = f"{user_id}:{timestamp}:{self.jwt_secret}"
        token = hashlib.sha256(data.encode()).hexdigest()
        return f"{timestamp}:{token}"
    
    def validate_csrf_token(self, token: str, user_id: str) -> bool:
        """Validate CSRF token."""
        try:
            timestamp_str, token_hash = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check if token is not too old (1 hour)
            if time.time() - timestamp > 3600:
                return False
            
            # Recreate expected token
            data = f"{user_id}:{timestamp_str}:{self.jwt_secret}"
            expected_hash = hashlib.sha256(data.encode()).hexdigest()
            
            return token_hash == expected_hash
            
        except (ValueError, AttributeError):
            return False
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics summary."""
        return {
            'blocked_ips_count': len(self.blocked_ips),
            'recent_events_count': len(self.suspicious_activities),
            'failed_attempts_count': sum(len(attempts) for attempts in self.failed_attempts.values()),
            'rate_limit_windows': len(self.rate_limiters),
            'last_security_scan': datetime.now().isoformat()
        }
    
    async def cleanup_security_data(self):
        """Clean up old security data."""
        try:
            # Clean up old failed attempts
            cutoff_time = datetime.now() - timedelta(hours=24)
            for ip in list(self.failed_attempts.keys()):
                self.failed_attempts[ip] = [
                    attempt for attempt in self.failed_attempts[ip] if attempt > cutoff_time
                ]
                if not self.failed_attempts[ip]:
                    del self.failed_attempts[ip]
            
            # Clean up old rate limit data
            current_time = time.time()
            for key in list(self.rate_limiters.keys()):
                request_times = self.rate_limiters[key]
                while request_times and request_times[0] < current_time - 3600:  # 1 hour
                    request_times.popleft()
                if not request_times:
                    del self.rate_limiters[key]
            
            # Keep only recent security events
            cutoff_time = datetime.now() - timedelta(days=7)
            self.suspicious_activities = [
                event for event in self.suspicious_activities if event.timestamp > cutoff_time
            ]
            
            self.logger.info("Security data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during security cleanup: {e}")
    
    async def close(self):
        """Close security manager connections."""
        try:
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("Security manager closed")
        except Exception as e:
            self.logger.error(f"Error closing security manager: {e}")


# Global security manager instance
security_manager = None

async def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global security_manager
    if security_manager is None:
        from production_config import get_production_config
        config = get_production_config()
        security_manager = SecurityManager(config)
        await security_manager.initialize()
    return security_manager

async def authenticate_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
    """FastAPI dependency for user authentication."""
    security = await get_security_manager()
    return await security.authenticate_request(credentials)