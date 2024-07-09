import sys

from pyrogram import Client, errors
from pyrogram.handlers import MessageHandler
from pyrogram.errors import RPCError
from pyrogram.errors import FloodWait, BadRequest, Forbidden, AuthKeyUnregistered, AuthKeyInvalid, AuthKeyDuplicated, SessionPasswordNeeded, PhoneNumberInvalid, PhoneNumberUnoccupied, PhoneNumberBanned, PhoneNumberFlood, UserIsBlocked, UserNotMutualContact, UserPrivacyRestricted, UserChannelsTooMuch, UsernameNotOccupied
import asyncio
from datetime import datetime
from pyrogram.types import (Message as PyrogramMessage)
from logging import Logger
from typing import Callable, Optional, List, Tuple
from math import floor
from sqlalchemy.orm import Session
from pyrogram.errors.exceptions.flood_420 import FloodWait
from asyncio import sleep
import random
from langdetect import detect
from nltk.tokenize import word_tokenize
import asyncio
import uvloop
from pyrogram import Client, compose
from pyrogram.errors import PeerIdInvalid, ChatAdminRequired
import nest_asyncio
from datetime import datetime
from typing import Union, Optional, AsyncGenerator