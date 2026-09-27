import enum
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Enum,
    ForeignKey,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.decl_api import DeclarativeBase


Base:DeclarativeBase = declarative_base()

class Account(Base):
    __tablename__ = 'accounts'
    account_id = Column(String, primary_key=True)
    account_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    users = relationship("User", back_populates="account")
    tickets = relationship("Ticket", back_populates="account")
    knowledge_articles = relationship("Knowledge", back_populates="account")

    def __repr__(self):
        return f"<Account(account_id='{self.account_id}', account_name='{self.account_name}')>"


class User(Base):
    __tablename__ = 'users'

    user_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    external_user_id = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    account = relationship("Account", back_populates="users")
    tickets = relationship("Ticket", back_populates="user")

    __table_args__ = (
        UniqueConstraint('account_id', 'external_user_id', name='uq_user_external_per_account'),
    )

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', user_name='{self.user_name}', external_user_id='{self.external_user_id}')>"



class Ticket(Base):
    __tablename__ = 'tickets'
    ticket_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    channel = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    account = relationship("Account", back_populates="tickets")
    user = relationship("User", back_populates="tickets")
    ticket_metadata = relationship("TicketMetadata", uselist=False, back_populates="ticket")
    messages = relationship("TicketMessage", back_populates="ticket")

    def __repr__(self):
        return f"<Ticket(ticket_id='{self.ticket_id}', channel='{self.channel}', created_at='{self.created_at}')>"


class TicketMetadata(Base):
    __tablename__ = 'ticket_metadata'
    ticket_id = Column(String, ForeignKey('tickets.ticket_id'), primary_key=True)
    status = Column(String, nullable=False)
    main_issue_type = Column(String)
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    ticket = relationship("Ticket", back_populates="ticket_metadata")

    def __repr__(self):
        return f"<TicketMetadata(ticket_id='{self.ticket_id}', status='{self.status}', issue_type='{self.main_issue_type}')>"


class RoleEnum(enum.Enum):
    user = "user"
    agent = "agent"
    ai = "ai"
    system = "system"


class TicketMessage(Base):
    __tablename__ = 'ticket_messages'
    message_id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey('tickets.ticket_id'), nullable=False)
    role = Column(Enum(RoleEnum, name="role_enum"), nullable=False)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    ticket = relationship("Ticket", back_populates="messages")

    def __repr__(self):
        short_content = (self.content[:30] + "...") if self.content and len(self.content) > 30 else self.content
        return f"<TicketMessage(message_id='{self.message_id}', role='{self.role.name}', content='{short_content}')>"


class Knowledge(Base):
    __tablename__ = 'knowledge'
    article_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    account = relationship("Account", back_populates="knowledge_articles")

    def __repr__(self):
        return f"<Knowledge(article_id='{self.article_id}', title='{self.title}')>"


# ---------------------------------------------------------------------------
# Tables added by the UDA-Hub agent system. The six tables above are the
# starter's; everything below supports memory, RAG caching, auditable logging
# and support actions that need a human approval (see agentic/design).
# ---------------------------------------------------------------------------

class CustomerMemory(Base):
    """Long-term memory: preferences and resolved issues that outlive a ticket."""
    __tablename__ = 'customer_memories'

    memory_id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey('accounts.account_id'), nullable=False)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    kind = Column(String, nullable=False)           # preference | resolved_issue | escalated_issue | fact
    key = Column(String)                            # preferences are upserted by key, e.g. "contact_channel"
    content = Column(Text, nullable=False)
    embedding = Column(Text)                        # JSON list[float]; NULL when embeddings are unavailable
    source_ticket_id = Column(String, ForeignKey('tickets.ticket_id'))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<CustomerMemory(kind='{self.kind}', key='{self.key}', content='{(self.content or '')[:40]}')>"


class KnowledgeEmbedding(Base):
    """Cached article embeddings, invalidated by content hash."""
    __tablename__ = 'knowledge_embeddings'

    article_id = Column(String, ForeignKey('knowledge.article_id'), primary_key=True)
    model = Column(String, primary_key=True)
    content_hash = Column(String, nullable=False)
    embedding = Column(Text, nullable=False)        # JSON list[float]
    created_at = Column(DateTime, default=datetime.now)


class AgentEvent(Base):
    """Structured, searchable audit log of every agent decision and tool call."""
    __tablename__ = 'agent_events'

    event_id = Column(String, primary_key=True)
    ts = Column(DateTime, default=datetime.now, index=True)
    run_id = Column(String, index=True)
    thread_id = Column(String, index=True)
    ticket_id = Column(String, index=True)
    agent = Column(String, index=True)
    event = Column(String, index=True)
    level = Column(String, default="INFO")
    payload = Column(Text)                          # JSON

    def __repr__(self):
        return f"<AgentEvent(agent='{self.agent}', event='{self.event}', ticket_id='{self.ticket_id}')>"


class SupportAction(Base):
    """Actions an agent may request but not complete alone (refunds, ...)."""
    __tablename__ = 'support_actions'

    action_id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey('tickets.ticket_id'))
    external_user_id = Column(String, nullable=False)
    action_type = Column(String, nullable=False)    # refund_request | ...
    status = Column(String, nullable=False)         # pending_approval | approved | rejected
    details = Column(Text)                          # JSON
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
