"""
Currency Converter Tool - Exchange rate and currency conversion operations
"""

from typing import Dict, Any, Optional, List
import logging
from ..utils.api_clients import ExchangeRateClient

logger = logging.getLogger(__name__)


class CurrencyConverter:
    """
    Utility tool for currency conversion operations.
    Uses ExchangeRate-API for real-time exchange rates.
    """
    
    def __init__(self, api_client: Optional[ExchangeRateClient] = None):
        """
        Initialize currency converter
        
        Args:
            api_client: Optional ExchangeRateClient instance for dependency injection
        """
        self.api_client = api_client or ExchangeRateClient()
        self.cached_rates: Dict[str, Dict[str, float]] = {}
        self.conversion_history: List[Dict[str, Any]] = []
    
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Get exchange rate between two currencies.
        
        Args:
            from_currency: Source currency code (e.g., 'USD')
            to_currency: Target currency code (e.g., 'EUR')
            
        Returns:
            Exchange rate or None if failed
        """
        if not from_currency or not to_currency:
            logger.error("Invalid currency codes provided")
            return None
        
        # Normalize currency codes
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Same currency
        if from_currency == to_currency:
            return 1.0
        
        # Check cache first
        cache_key = f"{from_currency}_{to_currency}"
        if from_currency in self.cached_rates and to_currency in self.cached_rates[from_currency]:
            rate = self.cached_rates[from_currency][to_currency]
            logger.info(f"Using cached rate {from_currency} -> {to_currency}: {rate}")
            return rate
        
        try:
            # Get rates from API
            rates_data = self.api_client.get_exchange_rates(from_currency)
            if not rates_data or "rates" not in rates_data:
                logger.error(f"Failed to get exchange rates for {from_currency}")
                return None
            
            rates = rates_data["rates"]
            if to_currency not in rates:
                logger.error(f"Exchange rate not available for {to_currency}")
                return None
            
            rate = rates[to_currency]
            
            # Cache the rates
            if from_currency not in self.cached_rates:
                self.cached_rates[from_currency] = {}
            self.cached_rates[from_currency].update(rates)
            
            logger.info(f"Retrieved exchange rate {from_currency} -> {to_currency}: {rate}")
            return rate
            
        except Exception as e:
            logger.error(f"Failed to get exchange rate {from_currency} -> {to_currency}: {e}")
            return None
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Convert amount from one currency to another.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Converted amount or None if failed
        """
        if not isinstance(amount, (int, float)) or amount < 0:
            logger.error(f"Invalid amount for conversion: {amount}")
            return None
        
        if amount == 0:
            return 0.0
        
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate is None:
            return None
        
        converted_amount = amount * rate
        
        # Record conversion
        self.conversion_history.append({
            "operation": "convert_amount",
            "input": {
                "amount": amount,
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "rate": rate
            },
            "result": converted_amount
        })
        
        logger.info(f"Converted {amount} {from_currency} -> {converted_amount:.2f} {to_currency} (rate: {rate})")
        return round(converted_amount, 2)
    
    def convert_cost_breakdown(self, costs: Dict[str, float], from_currency: str, to_currency: str) -> Dict[str, float]:
        """
        Convert entire cost breakdown to target currency.
        
        Args:
            costs: Dictionary of cost categories and amounts
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Dictionary with converted amounts
        """
        if not costs:
            return {}
        
        converted_costs = {}
        
        for category, amount in costs.items():
            if amount is None or not isinstance(amount, (int, float)):
                logger.warning(f"Invalid cost amount for category '{category}': {amount}")
                converted_costs[category] = 0.0
                continue
            
            converted_amount = self.convert_amount(amount, from_currency, to_currency)
            if converted_amount is not None:
                converted_costs[category] = converted_amount
            else:
                logger.warning(f"Failed to convert amount for category '{category}'")
                converted_costs[category] = amount  # Fallback to original amount
        
        # Record bulk conversion
        self.conversion_history.append({
            "operation": "convert_cost_breakdown",
            "input": {
                "costs": costs,
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper()
            },
            "result": converted_costs
        })
        
        logger.info(f"Converted {len(costs)} cost categories from {from_currency} to {to_currency}")
        return converted_costs
    
    def get_supported_currencies(self) -> Optional[List[str]]:
        """
        Get list of supported currency codes.
        
        Returns:
            List of currency codes or None if failed
        """
        try:
            currencies = self.api_client.get_supported_currencies()
            if currencies and "supported_codes" in currencies:
                # Extract currency codes from the supported_codes list
                codes = [code[0] for code in currencies["supported_codes"]]
                logger.info(f"Retrieved {len(codes)} supported currencies")
                return codes
            else:
                logger.error("Failed to get supported currencies")
                return None
        except Exception as e:
            logger.error(f"Error getting supported currencies: {e}")
            return None
    
    def is_currency_supported(self, currency_code: str) -> bool:
        """
        Check if a currency code is supported.
        
        Args:
            currency_code: Currency code to check
            
        Returns:
            True if supported, False otherwise
        """
        if not currency_code:
            return False
        
        supported = self.get_supported_currencies()
        if supported is None:
            # Fallback to common currencies if API fails
            common_currencies = [
                'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'SEK', 'NZD',
                'MXN', 'SGD', 'HKD', 'NOK', 'INR', 'KRW', 'TRY', 'BRL', 'ZAR', 'RUB'
            ]
            return currency_code.upper() in common_currencies
        
        return currency_code.upper() in supported
    
    def get_conversion_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all conversions performed.
        
        Returns:
            List of conversion records
        """
        return self.conversion_history.copy()
    
    def clear_cache(self):
        """Clear cached exchange rates."""
        self.cached_rates.clear()
        logger.info("Exchange rate cache cleared")
    
    def clear_history(self):
        """Clear conversion history."""
        self.conversion_history.clear()
        logger.info("Conversion history cleared")