// ag-grid-enterprise v4.2.10
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var main_1 = require("ag-grid/main");
var AbstractColumnDropPanel = (function (_super) {
    __extends(AbstractColumnDropPanel, _super);
    function AbstractColumnDropPanel(horizontal) {
        _super.call(this, "<div class=\"ag-column-drop ag-font-style ag-column-drop-" + (horizontal ? 'horizontal' : 'vertical') + "\"></div>");
        this.guiDestroyFunctions = [];
        this.horizontal = horizontal;
    }
    AbstractColumnDropPanel.prototype.setBeans = function (beans) {
        this.beans = beans;
    };
    AbstractColumnDropPanel.prototype.destroy = function () {
        this.destroyGui();
        _super.prototype.destroy.call(this);
    };
    AbstractColumnDropPanel.prototype.destroyGui = function () {
        this.guiDestroyFunctions.forEach(function (func) { return func(); });
        this.guiDestroyFunctions.length = 0;
        main_1.Utils.removeAllChildren(this.getGui());
    };
    AbstractColumnDropPanel.prototype.init = function (params) {
        this.params = params;
        this.logger = this.beans.loggerFactory.create('AbstractColumnDropPanel');
        this.beans.eventService.addEventListener(main_1.Events.EVENT_COLUMN_EVERYTHING_CHANGED, this.refreshGui.bind(this));
        this.setupDropTarget();
        // we don't know if this bean will be initialised before columnController.
        // if columnController first, then below will work
        // if columnController second, then below will put blank in, and then above event gets first when columnController is set up
        this.refreshGui();
    };
    AbstractColumnDropPanel.prototype.setupDropTarget = function () {
        this.dropTarget = {
            eContainer: this.getGui(),
            onDragging: this.onDragging.bind(this),
            onDragEnter: this.onDragEnter.bind(this),
            onDragLeave: this.onDragLeave.bind(this),
            onDragStop: this.onDragStop.bind(this)
        };
        this.beans.dragAndDropService.addDropTarget(this.dropTarget);
    };
    AbstractColumnDropPanel.prototype.onDragging = function () {
    };
    AbstractColumnDropPanel.prototype.onDragEnter = function (draggingEvent) {
        // this will contain all columns that are potential drops
        var dragColumns = draggingEvent.dragSource.dragItem;
        // take out columns that are not groupable
        var goodDragColumns = main_1.Utils.filter(dragColumns, this.isColumnDroppable.bind(this));
        var weHaveColumnsToDrag = goodDragColumns.length > 0;
        if (weHaveColumnsToDrag) {
            this.potentialDndColumns = goodDragColumns;
            this.beans.dragAndDropService.setGhostIcon(this.params.dragAndDropIcon);
            this.refreshGui();
        }
        else {
            this.beans.dragAndDropService.setGhostIcon(null);
        }
    };
    AbstractColumnDropPanel.prototype.onDragLeave = function (draggingEvent) {
        // if the dragging started from us, we remove the group, however if it started
        // someplace else, then we don't, as it was only 'asking'
        var thisPanelStartedTheDrag = draggingEvent.dragSource.dragSourceDropTarget === this.dropTarget;
        if (thisPanelStartedTheDrag) {
            var columns = draggingEvent.dragSource.dragItem;
            this.removeColumns(columns);
        }
        if (this.potentialDndColumns) {
            this.potentialDndColumns = null;
            this.refreshGui();
        }
    };
    AbstractColumnDropPanel.prototype.onDragStop = function () {
        if (this.potentialDndColumns) {
            this.addColumns(this.potentialDndColumns);
            this.potentialDndColumns = null;
            this.refreshGui();
        }
    };
    AbstractColumnDropPanel.prototype.refreshGui = function () {
        this.destroyGui();
        this.addIconAndTitleToGui();
        this.addEmptyMessageToGui();
        this.addExistingColumnsToGui();
        this.addPotentialDragItemsToGui();
    };
    AbstractColumnDropPanel.prototype.addPotentialDragItemsToGui = function () {
        var _this = this;
        var first = this.isExistingColumnsEmpty();
        if (this.potentialDndColumns) {
            this.potentialDndColumns.forEach(function (column) {
                if (!first) {
                    _this.addArrowToGui();
                }
                first = false;
                var ghostCell = new ColumnComponent(column, _this.dropTarget, true);
                ghostCell.addEventListener(ColumnComponent.EVENT_COLUMN_REMOVE, _this.removeColumns.bind(_this, [column]));
                _this.beans.context.wireBean(ghostCell);
                _this.getGui().appendChild(ghostCell.getGui());
                _this.guiDestroyFunctions.push(function () { return ghostCell.destroy(); });
            });
        }
    };
    AbstractColumnDropPanel.prototype.addExistingColumnsToGui = function () {
        var _this = this;
        var existingColumns = this.getExistingColumns();
        existingColumns.forEach(function (column, index) {
            if (index > 0) {
                _this.addArrowToGui();
            }
            var cell = new ColumnComponent(column, _this.dropTarget, false);
            cell.addEventListener(ColumnComponent.EVENT_COLUMN_REMOVE, _this.removeColumns.bind(_this, [column]));
            _this.beans.context.wireBean(cell);
            _this.getGui().appendChild(cell.getGui());
            _this.guiDestroyFunctions.push(function () { return cell.destroy(); });
        });
    };
    AbstractColumnDropPanel.prototype.addIconAndTitleToGui = function () {
        var iconFaded = this.horizontal && this.isExistingColumnsEmpty();
        var eGroupIcon = this.params.iconFactory();
        main_1.Utils.addCssClass(eGroupIcon, 'ag-column-drop-icon');
        main_1.Utils.addOrRemoveCssClass(eGroupIcon, 'ag-faded', iconFaded);
        this.getGui().appendChild(eGroupIcon);
        if (!this.horizontal) {
            var eTitle = document.createElement('span');
            eTitle.innerHTML = this.params.title;
            main_1.Utils.addCssClass(eTitle, 'ag-column-drop-title');
            main_1.Utils.addOrRemoveCssClass(eTitle, 'ag-faded', iconFaded);
            this.getGui().appendChild(eTitle);
        }
    };
    AbstractColumnDropPanel.prototype.isExistingColumnsEmpty = function () {
        return this.getExistingColumns().length === 0;
    };
    AbstractColumnDropPanel.prototype.addEmptyMessageToGui = function () {
        var showEmptyMessage = this.isExistingColumnsEmpty() && !this.potentialDndColumns;
        if (!showEmptyMessage) {
            return;
        }
        var eMessage = document.createElement('span');
        eMessage.innerHTML = this.params.emptyMessage;
        main_1.Utils.addCssClass(eMessage, 'ag-column-drop-empty-message');
        this.getGui().appendChild(eMessage);
    };
    AbstractColumnDropPanel.prototype.addArrowToGui = function () {
        // only add the arrows if the layout is horizontal
        if (this.horizontal) {
            var eArrow = document.createElement('span');
            eArrow.innerHTML = '&#8594;';
            this.getGui().appendChild(eArrow);
        }
    };
    return AbstractColumnDropPanel;
})(main_1.Component);
exports.AbstractColumnDropPanel = AbstractColumnDropPanel;
var ColumnComponent = (function (_super) {
    __extends(ColumnComponent, _super);
    function ColumnComponent(column, dragSourceDropTarget, ghost) {
        _super.call(this, ColumnComponent.TEMPLATE);
        this.column = column;
        this.dragSourceDropTarget = dragSourceDropTarget;
        this.ghost = ghost;
    }
    ColumnComponent.prototype.init = function () {
        this.displayName = this.columnController.getDisplayNameForCol(this.column);
        this.setupComponents();
        if (!this.ghost) {
            this.addDragSource();
        }
    };
    ColumnComponent.prototype.addDragSource = function () {
        var dragSource = {
            eElement: this.getGui(),
            dragItem: [this.column],
            dragItemName: this.displayName,
            dragSourceDropTarget: this.dragSourceDropTarget
        };
        this.dragAndDropService.addDragSource(dragSource);
    };
    ColumnComponent.prototype.setupComponents = function () {
        var _this = this;
        var eText = this.getGui().querySelector('#eText');
        var btRemove = this.getGui().querySelector('#btRemove');
        eText.innerHTML = this.displayName;
        this.addDestroyableEventListener(btRemove, 'click', function () { return _this.dispatchEvent(ColumnComponent.EVENT_COLUMN_REMOVE); });
        if (this.ghost) {
            main_1.Utils.addCssClass(this.getGui(), 'ag-column-drop-cell-ghost');
        }
    };
    ColumnComponent.EVENT_COLUMN_REMOVE = 'columnRemove';
    ColumnComponent.TEMPLATE = '<span class="ag-column-drop-cell">' +
        '<span id="eText" class="ag-column-drop-cell-text"></span>' +
        '<span id="btRemove" class="ag-column-drop-cell-button">&#10006;</span>' +
        '</span>';
    __decorate([
        main_1.Autowired('dragAndDropService'), 
        __metadata('design:type', main_1.DragAndDropService)
    ], ColumnComponent.prototype, "dragAndDropService", void 0);
    __decorate([
        main_1.Autowired('columnController'), 
        __metadata('design:type', main_1.ColumnController)
    ], ColumnComponent.prototype, "columnController", void 0);
    __decorate([
        main_1.Autowired('gridPanel'), 
        __metadata('design:type', main_1.GridPanel)
    ], ColumnComponent.prototype, "gridPanel", void 0);
    __decorate([
        main_1.PostConstruct, 
        __metadata('design:type', Function), 
        __metadata('design:paramtypes', []), 
        __metadata('design:returntype', void 0)
    ], ColumnComponent.prototype, "init", null);
    return ColumnComponent;
})(main_1.Component);
