"""
Workflow Automation Engine
Multi-step workflow execution with conditional logic and error handling.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowEngine:
    """
    Executes multi-step business workflows with conditional logic.
    """
    
    def __init__(self):
        self.workflows = {}
        self.executions = {}
        self.execution_counter = 0
        
    def define_workflow(self, name: str, steps: List[Dict]) -> str:
        """
        Define a new workflow.
        
        Args:
            name: Workflow name
            steps: List of step definitions
            
        Returns:
            Workflow ID
        """
        workflow_id = f"wf_{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.workflows[workflow_id] = {
            'id': workflow_id,
            'name': name,
            'steps': steps,
            'created_at': datetime.now().isoformat(),
            'executions': 0
        }
        
        return workflow_id
    
    def execute_workflow(self, workflow_id: str, context: Dict = None) -> Dict:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow to execute
            context: Initial context data
            
        Returns:
            Execution result
        """
        if workflow_id not in self.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflows[workflow_id]
        execution_id = self._create_execution(workflow_id)
        
        execution = {
            'id': execution_id,
            'workflow_id': workflow_id,
            'status': WorkflowStatus.RUNNING,
            'started_at': datetime.now(),
            'context': context or {},
            'step_results': [],
            'current_step': 0
        }
        
        self.executions[execution_id] = execution
        
        try:
            for i, step in enumerate(workflow['steps']):
                execution['current_step'] = i
                
                # Check condition
                if not self._check_condition(step.get('condition'), execution['context']):
                    execution['step_results'].append({
                        'step': i,
                        'name': step['name'],
                        'status': 'skipped',
                        'reason': 'condition_not_met'
                    })
                    continue
                
                # Execute step
                result = self._execute_step(step, execution['context'])
                
                execution['step_results'].append({
                    'step': i,
                    'name': step['name'],
                    'status': 'completed' if result.get('success') else 'failed',
                    'result': result
                })
                
                # Update context with step output
                if result.get('output'):
                    execution['context'].update(result['output'])
                
                # Check if step failed and should stop workflow
                if not result.get('success') and step.get('stop_on_error', True):
                    execution['status'] = WorkflowStatus.FAILED
                    break
            
            if execution['status'] == WorkflowStatus.RUNNING:
                execution['status'] = WorkflowStatus.COMPLETED
            
        except Exception as e:
            execution['status'] = WorkflowStatus.FAILED
            execution['error'] = str(e)
        
        execution['completed_at'] = datetime.now()
        execution['duration_seconds'] = (execution['completed_at'] - execution['started_at']).seconds
        
        # Update workflow stats
        workflow['executions'] += 1
        
        return execution
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get status of a workflow execution."""
        return self.executions.get(execution_id)
    
    def pause_execution(self, execution_id: str) -> bool:
        """Pause a running workflow execution."""
        if execution_id in self.executions:
            self.executions[execution_id]['status'] = WorkflowStatus.PAUSED
            return True
        return False
    
    def resume_execution(self, execution_id: str) -> Dict:
        """Resume a paused workflow execution."""
        if execution_id not in self.executions:
            return {'error': 'Execution not found'}
        
        execution = self.executions[execution_id]
        if execution['status'] != WorkflowStatus.PAUSED:
            return {'error': 'Execution is not paused'}
        
        execution['status'] = WorkflowStatus.RUNNING
        
        # Continue from current step
        workflow = self.workflows[execution['workflow_id']]
        remaining_steps = workflow['steps'][execution['current_step']:]
        
        # Execute remaining steps (simplified - would need full execution logic)
        return execution
    
    def _execute_step(self, step: Dict, context: Dict) -> Dict:
        """Execute a single workflow step."""
        step_type = step.get('type')
        
        if step_type == 'action':
            return self._execute_action(step, context)
        elif step_type == 'decision':
            return self._execute_decision(step, context)
        elif step_type == 'loop':
            return self._execute_loop(step, context)
        elif step_type == 'wait':
            return self._execute_wait(step, context)
        else:
            return {'success': False, 'error': f'Unknown step type: {step_type}'}
    
    def _execute_action(self, step: Dict, context: Dict) -> Dict:
        """Execute an action step."""
        action = step.get('action')
        params = step.get('params', {})
        
        # Resolve parameters from context
        resolved_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                context_key = value[2:-1]
                resolved_params[key] = context.get(context_key, value)
            else:
                resolved_params[key] = value
        
        # Execute action (would call actual functions in production)
        result = {
            'success': True,
            'output': {f'{action}_result': f'Executed {action} with {resolved_params}'}
        }
        
        return result
    
    def _execute_decision(self, step: Dict, context: Dict) -> Dict:
        """Execute a decision step."""
        condition = step.get('condition')
        
        if self._check_condition(condition, context):
            return {'success': True, 'output': {'decision':  'true_branch'}}
        else:
            return {'success': True, 'output': {'decision': 'false_branch'}}
    
    def _execute_loop(self, step: Dict, context: Dict) -> Dict:
        """Execute a loop step."""
        iterations = step.get('iterations', 1)
        results = []
        
        for i in range(iterations):
            # Would execute loop body here
            results.append(f'iteration_{i}')
        
        return {'success': True, 'output': {'loop_results': results}}
    
    def _execute_wait(self, step: Dict, context: Dict) -> Dict:
        """Execute a wait/delay step."""
        # In production, would actually wait
        duration = step.get('duration', 0)
        return {'success': True, 'output': {'waited': duration}}
    
    def _check_condition(self, condition: Optional[str], context: Dict) -> bool:
        """Check if a condition is met."""
        if not condition:
            return True
        
        # Simple condition evaluation (would use safer eval in production)
        try:
            # Replace context variables
            for key, value in context.items():
                condition = condition.replace(f'${{{key}}}', str(value))
            
            # Very simplified - in production would use ast.literal_eval or safe expression parser
            if '>' in condition:
                parts = condition.split('>')
                return float(parts[0].strip()) > float(parts[1].strip())
            elif '<' in condition:
                parts = condition.split('<')
                return float(parts[0].strip()) < float(parts[1].strip())
            elif '==' in condition:
                parts = condition.split('==')
                return parts[0].strip() == parts[1].strip()
            else:
                return bool(condition)
        except:
            return True
    
    def _create_execution(self, workflow_id: str) -> str:
        """Create new execution ID."""
        self.execution_counter += 1
        return f"exec_{workflow_id}_{self.execution_counter}"


class TaskScheduler:
    """
    Schedules and executes recurring tasks.
    """
    
    def __init__(self):
        self.tasks = {}
        self.task_counter = 0
        self.execution_log = []
        
    def schedule_task(self, name: str, schedule: str, action: str, params: Dict = None) -> str:
        """
        Schedule a recurring task.
        
        Args:
            name: Task name
            schedule: Cron-like schedule (simplified)
            action: Action to execute
            params: Action parameters
            
        Returns:
            Task ID
        """
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        self.tasks[task_id] = {
            'id': task_id,
            'name': name,
            'schedule': schedule,
            'action': action,
            'params': params or {},
            'enabled': True,
            'last_run': None,
            'next_run': self._calculate_next_run(schedule),
            'run_count': 0
        }
        
        return task_id
    
    def run_due_tasks(self) -> List[Dict]:
        """Run all tasks that are due."""
        now = datetime.now()
        results = []
        
        for task_id, task in self.tasks.items():
            if not task['enabled']:
                continue
            
            if task['next_run'] and task['next_run'] <= now:
                result = self._execute_task(task)
                results.append(result)
                
                # Update task
                task['last_run'] = now
                task['next_run'] = self._calculate_next_run(task['schedule'], now)
                task['run_count'] += 1
        
        return results
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id]['enabled'] = True
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id]['enabled'] = False
            return True
        return False
    
    def get_scheduled_tasks(self) -> List[Dict]:
        """Get all scheduled tasks."""
        return list(self.tasks.values())
    
    def _execute_task(self, task: Dict) -> Dict:
        """Execute a scheduled task."""
        start_time = datetime.now()
        
        try:
            # Execute action (would call actual function in production)
            result = {
                'task_id': task['id'],
                'status': 'success',
                'executed_at': start_time.isoformat(),
                'action': task['action']
            }
            
            self.execution_log.append(result)
            
        except Exception as e:
            result = {
                'task_id': task['id'],
                'status': 'failed',
                'executed_at': start_time.isoformat(),
                'error': str(e)
            }
        
        return result
    
    def _calculate_next_run(self, schedule: str, from_time: datetime = None) -> datetime:
        """Calculate next run time based on schedule."""
        if not from_time:
            from_time = datetime.now()
        
        # Simplified schedule parsing
        if schedule == 'daily':
            return from_time + timedelta(days=1)
        elif schedule == 'hourly':
            return from_time + timedelta(hours=1)
        elif schedule == 'weekly':
            return from_time + timedelta(weeks=1)
        elif schedule == 'monthly':
            return from_time + timedelta(days=30)
        elif schedule.startswith('every_'):
            # Format: "every_Nh" (every N hours) or "every_Nd" (every N days)
            parts = schedule.split('_')
            if len(parts) == 2:
                value = int(parts[1][:-1])
                unit = parts[1][-1]
                
                if unit == 'h':
                    return from_time + timedelta(hours=value)
                elif unit == 'd':
                    return from_time + timedelta(days=value)
        
        # Default: next hour
        return from_time + timedelta(hours=1)


class ApprovalWorkflow:
    """
    Manages approval workflows and requests.
    """
    
    def __init__(self):
        self.approvals = {}
        self.approval_counter = 0
        
    def create_approval_request(self, title: str, details: Dict, approvers: List[str], required_approvals: int = 1) -> str:
        """Create a new approval request."""
        self.approval_counter += 1
        approval_id = f"approval_{self.approval_counter}"
        
        self.approvals[approval_id] = {
            'id': approval_id,
            'title': title,
            'details': details,
            'approvers': approvers,
            'required_approvals': required_approvals,
            'status': 'pending',
            'approvals': [],
            'rejections': [],
            'created_at': datetime.now().isoformat()
        }
        
        return approval_id
    
    def approve(self, approval_id: str, approver: str, comments: str = '') -> Dict:
        """Approve a request."""
        if approval_id not in self.approvals:
            return {'error': 'Approval request not found'}
        
        approval = self.approvals[approval_id]
        
        if approver not in approval['approvers']:
            return {'error': 'User not authorized to approve'}
        
        if approver in [a['approver'] for a in approval['approvals']]:
            return {'error': 'User already approved'}
        
        approval['approvals'].append({
            'approver': approver,
            'timestamp': datetime.now().isoformat(),
            'comments': comments
        })
        
        # Check if enough approvals
        if len(approval['approvals']) >= approval['required_approvals']:
            approval['status'] = 'approved'
            approval['approved_at'] = datetime.now().isoformat()
        
        return approval
    
    def reject(self, approval_id: str, approver: str, reason: str) -> Dict:
        """Reject a request."""
        if approval_id not in self.approvals:
            return {'error': 'Approval request not found'}
        
        approval = self.approvals[approval_id]
        
        if approver not in approval['approvers']:
            return {'error': 'User not authorized to reject'}
        
        approval['rejections'].append({
            'approver': approver,
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        })
        
        approval['status'] = 'rejected'
        approval['rejected_at'] = datetime.now().isoformat()
        
        return approval
    
    def get_pending_approvals(self, approver: str = None) -> List[Dict]:
        """Get pending approval requests."""
        pending = [a for a in self.approvals.values() if a['status'] == 'pending']
        
        if approver:
            pending = [a for a in pending if approver in a['approvers']]
        
        return pending
